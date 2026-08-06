"""Market research agent: Tavily search + LLM synthesis, grounding the calendar
multipliers that today are either hand-typed (app/core/items.py) or an ungrounded
LLM guess with no real market data behind it (market_priors.py's generate_priors_llm).

WHY THIS EXISTS
---------------
Every multiplier in this system -- "Ramadan quadruples konafa demand", "Sham
El-Nessim triples feteer" -- is currently either a domain expert's typed-in guess or
an LLM asked to guess with nothing but a menu list, no real information about the
actual market. This agent replaces the guess with a grounded one: search for real
sources about how a category actually behaves around a real event, then ask the LLM
to estimate a multiplier FROM those sources, citing which ones it used.

GUARDRAILS (do not remove without re-reading plan.md Part C)
--------------------------------------------------------------
- Never writes directly into a live restaurant's priors. Every result is
  "researched"/"rejected"/"failed" -- storage and the pending_review -> approved
  gate live in app/integration/priors_store.py, not here.
- Every accepted multiplier is re-validated through market_priors._clean() -- the
  exact same bounded-range check the hand-typed/ungrounded-LLM path already uses.
  Web search results are untrusted external content; a search result is not ground
  truth just because it has a URL attached (OWASP LLM08, RAG Data Poisoning).
- Bounded search count per run (`max_searches_per_run`) -- same DoS discipline as
  the request-size limits added to the API schemas (OWASP LLM04).
- Every external call (Tavily, LLM) is wrapped so a failure degrades to a
  "failed"/"unavailable" result for that one pair, never an exception that takes
  down the whole batch -- same fallback discipline as LLMGenerator/generate_priors_llm.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from app.core.market_priors import KEY_TO_ATTR, _clean

DEFAULT_LOCATION = "Egypt"
MAX_SEARCHES_PER_RUN = 30  # bounded: see module docstring


@dataclass
class MarketResearchAgent:
    tavily_api_key: str | None = None
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    llm_model: str | None = None
    max_searches_per_run: int = MAX_SEARCHES_PER_RUN
    timeout: float = 20.0

    def __post_init__(self) -> None:
        self.tavily_api_key = self.tavily_api_key or os.getenv("TAVILY_API_KEY")
        self.llm_api_key = self.llm_api_key or os.getenv("LLM_API_KEY")
        self.llm_base_url = self.llm_base_url or os.getenv(
            "LLM_BASE_URL", "https://api.groq.com/openai/v1"
        )
        self.llm_model = self.llm_model or os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

    @property
    def available(self) -> bool:
        """Both keys are required -- researching without a search backend is just
        the old ungrounded guess again, and synthesising without an LLM leaves raw
        search results with no multiplier at all."""
        return bool(self.tavily_api_key and self.llm_api_key)

    # -- Tavily -----------------------------------------------------------------

    def _search(self, query: str) -> tuple[list[dict], str | None]:
        from tavily import TavilyClient

        client = TavilyClient(api_key=self.tavily_api_key)
        result = client.search(
            query, search_depth="basic", max_results=4, include_answer=True,
        )
        sources = [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                # Truncated: this goes straight into the LLM prompt next: capping it
                # bounds prompt size the same way schemas.py's max_length fields do
                # for API inputs (OWASP LLM04).
                "content": (r.get("content") or "")[:500],
            }
            for r in result.get("results", [])
        ]
        return sources, result.get("answer")

    # -- LLM synthesis ------------------------------------------------------------

    def _prompt(self, category: str, event: str, location: str,
                sources: list[dict], answer: str | None) -> str:
        context = "\n\n".join(
            f"- {s['title']} ({s['url']}):\n  {s['content']}" for s in sources
        )
        return (
            f"You are estimating how much demand for a bakery/restaurant category "
            f"changes around a specific calendar event, based ONLY on the sources "
            f"below -- not on general knowledge.\n\n"
            f"Category: {category}\n"
            f"Event: {event}\n"
            f"Location: {location}\n\n"
            f"Search summary: {answer or '(none)'}\n\n"
            f"Sources:\n{context}\n\n"
            f"Return JSON only, this exact shape:\n"
            f'{{"multiplier": <number>, "confidence": "low"|"medium"|"high", '
            f'"reasoning": "<one sentence, must reference what the sources actually say>"}}\n\n'
            f"multiplier = demand during the event / normal demand (1.0 = no change, "
            f"2.0 = double). If the sources don't actually support an estimate for "
            f"this specific category+event, return {{\"multiplier\": 1.0, "
            f'"confidence": "low", "reasoning": "sources did not cover this pair"}}.'
        )

    def _synthesize(self, category: str, event: str, location: str,
                     sources: list[dict], answer: str | None) -> dict:
        import httpx

        resp = httpx.post(
            f"{self.llm_base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.llm_api_key}"},
            json={
                "model": self.llm_model,
                "messages": [
                    {"role": "user", "content": self._prompt(category, event, location, sources, answer)}
                ],
                "temperature": 0.2,  # low: this is estimation from evidence, not creative copy
                "response_format": {"type": "json_object"},
                "max_tokens": 300,
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return json.loads(resp.json()["choices"][0]["message"]["content"])

    # -- Public API ---------------------------------------------------------------

    def research_one(self, category: str, event: str, location: str = DEFAULT_LOCATION) -> dict:
        """Research a single (category, event) pair. Never raises -- every failure
        mode returns a dict with `status` explaining what happened, so a caller can
        research a whole menu without one bad pair aborting the rest."""
        base = {"category": category, "event": event}

        if not self.available:
            return {**base, "status": "unavailable",
                    "reason": "TAVILY_API_KEY or LLM_API_KEY not configured"}
        if event not in KEY_TO_ATTR:
            return {**base, "status": "rejected", "reason": f"unknown event key {event!r}"}

        try:
            query = f"{category} bakery sales {event} {location} demand change statistics"
            sources, answer = self._search(query)
        except Exception as exc:
            return {**base, "status": "search_failed", "reason": str(exc)}

        if not sources:
            return {**base, "status": "no_sources"}

        try:
            payload = self._synthesize(category, event, location, sources, answer)
        except Exception as exc:
            return {**base, "status": "synthesis_failed", "reason": str(exc)}

        cleaned = _clean({event: payload.get("multiplier")})
        if event not in cleaned:
            return {**base, "status": "rejected",
                    "reason": "multiplier missing or outside the valid range after validation"}

        return {
            **base,
            "status": "researched",
            "multiplier": cleaned[event],
            "confidence": payload.get("confidence", "low"),
            "reasoning": payload.get("reasoning", ""),
            "sources": [s["url"] for s in sources if s["url"]],
        }

    def research_categories(
        self, categories: list[str], events: list[str] | None = None,
        location: str = DEFAULT_LOCATION,
    ) -> list[dict]:
        """Research every (category, event) pair, bounded to `max_searches_per_run`
        total searches regardless of how many categories/events are passed in."""
        events = events or list(KEY_TO_ATTR)
        pairs = [(c, e) for c in categories for e in events][: self.max_searches_per_run]
        return [self.research_one(c, e, location) for c, e in pairs]
