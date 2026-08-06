"""Market research agent (plan.md Part C): Tavily search + LLM synthesis, and the
pending_review -> approved gate that keeps a researched multiplier out of live
forecasts until explicitly accepted.

No real TAVILY_API_KEY/LLM_API_KEY in this environment -- the search/synthesis
methods are monkeypatched to verify the actual parsing/validation/bounding logic
works, the same way test_market_priors.py tests the fallback path for the existing
LLM without a real key. The "unavailable" (no keys at all) path is tested for real,
since it needs no network access.
"""

from __future__ import annotations

import pytest

from app.agents.market_research import MarketResearchAgent


def test_unavailable_without_both_keys(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    agent = MarketResearchAgent()
    assert agent.available is False

    result = agent.research_one("bread", "ramadan")
    assert result["status"] == "unavailable"


def test_available_requires_tavily_and_llm_key():
    assert MarketResearchAgent(tavily_api_key="t", llm_api_key=None).available is False
    assert MarketResearchAgent(tavily_api_key=None, llm_api_key="l").available is False
    assert MarketResearchAgent(tavily_api_key="t", llm_api_key="l").available is True


def test_unknown_event_key_is_rejected_before_any_network_call():
    agent = MarketResearchAgent(tavily_api_key="t", llm_api_key="l")
    result = agent.research_one("bread", "not_a_real_event")
    assert result["status"] == "rejected"


def _agent_with_fakes(monkeypatch, sources, answer, llm_payload):
    agent = MarketResearchAgent(tavily_api_key="t", llm_api_key="l")
    monkeypatch.setattr(agent, "_search", lambda query: (sources, answer))
    monkeypatch.setattr(agent, "_synthesize", lambda *a, **k: llm_payload)
    return agent


def test_a_valid_researched_multiplier_is_accepted(monkeypatch):
    sources = [{"title": "Egypt Retail Report", "url": "https://example.com/a", "content": "..."}]
    agent = _agent_with_fakes(
        monkeypatch, sources, "Ramadan sweets demand roughly doubles",
        {"multiplier": 2.1, "confidence": "medium", "reasoning": "sources report ~2x demand"},
    )
    result = agent.research_one("sweet", "ramadan")
    assert result["status"] == "researched"
    assert result["multiplier"] == 2.1
    assert result["sources"] == ["https://example.com/a"]
    assert "reasoning" in result


def test_out_of_bounds_multiplier_is_rejected_not_silently_clamped(monkeypatch):
    """market_priors._clean()'s MIN_MULT/MAX_MULT bound is reused here -- a
    hallucinated 500x must not become a stored result at all (OWASP LLM08)."""
    sources = [{"title": "X", "url": "https://example.com/b", "content": "..."}]
    agent = _agent_with_fakes(
        monkeypatch, sources, None,
        {"multiplier": 500.0, "confidence": "low", "reasoning": "..."},
    )
    result = agent.research_one("bread", "weekend")
    assert result["status"] == "rejected"


def test_no_sources_found_is_reported_not_faked(monkeypatch):
    agent = _agent_with_fakes(monkeypatch, [], None, {})
    result = agent.research_one("bread", "eid")
    assert result["status"] == "no_sources"


def test_search_failure_does_not_raise(monkeypatch):
    agent = MarketResearchAgent(tavily_api_key="t", llm_api_key="l")

    def boom(query):
        raise ConnectionError("network unreachable")

    monkeypatch.setattr(agent, "_search", boom)
    result = agent.research_one("bread", "ramadan")
    assert result["status"] == "search_failed"


def test_research_categories_is_bounded(monkeypatch):
    """5 categories x 9 events = 45 pairs; must not exceed max_searches_per_run."""
    agent = MarketResearchAgent(tavily_api_key="t", llm_api_key="l", max_searches_per_run=10)
    monkeypatch.setattr(agent, "research_one", lambda c, e, location="Egypt": {"status": "unavailable"})

    results = agent.research_categories(["bread", "pastry", "cake", "sweet", "savoury"])
    assert len(results) == 10


# -- PriorsStore: the pending_review -> approved gate ----------------------------------

MONGO_TEST_URL = "mongodb://127.0.0.1:27017/prediction_model_test"


def _mongo_reachable() -> bool:
    try:
        from pymongo import MongoClient

        MongoClient(MONGO_TEST_URL, serverSelectionTimeoutMS=1000).admin.command("ping")
        return True
    except Exception:
        return False


requires_mongo = pytest.mark.skipif(
    not _mongo_reachable(), reason="no local MongoDB reachable at 127.0.0.1:27017"
)


@pytest.fixture
def priors_store():
    from pymongo import MongoClient

    from app.integration.priors_store import PriorsStore

    MongoClient(MONGO_TEST_URL).get_default_database()["ai_researched_priors"].delete_many({})
    store = PriorsStore(MONGO_TEST_URL)
    yield store
    MongoClient(MONGO_TEST_URL).get_default_database()["ai_researched_priors"].delete_many({})


@requires_mongo
def test_researched_result_is_pending_not_live(priors_store):
    """The whole point of the gate: a stored research result must not be readable
    as an approved multiplier until it's explicitly approved."""
    priors_store.save_researched("R1", {
        "category": "sweet", "event": "ramadan", "status": "researched",
        "multiplier": 2.5, "confidence": "medium", "reasoning": "...", "sources": ["https://x"],
    })
    assert priors_store.approved_multipliers("R1", "sweet") == {}
    pending = priors_store.list_pending("R1")
    assert len(pending) == 1
    assert pending[0]["reviewStatus"] == "pending_review"


@requires_mongo
def test_approve_moves_it_into_the_live_path(priors_store):
    priors_store.save_researched("R1", {
        "category": "sweet", "event": "ramadan", "status": "researched",
        "multiplier": 2.5, "confidence": "medium", "reasoning": "...", "sources": [],
    })
    assert priors_store.approve("R1", "sweet", "ramadan") is True
    assert priors_store.approved_multipliers("R1", "sweet") == {"ramadan": 2.5}
    assert priors_store.list_pending("R1") == []


@requires_mongo
def test_approving_nonexistent_research_reports_failure(priors_store):
    assert priors_store.approve("R1", "bread", "weekend") is False


@requires_mongo
def test_re_researching_does_not_undo_an_approval(priors_store):
    """Re-running research must not silently overwrite an explicit approval with a
    fresh, unreviewed number."""
    priors_store.save_researched("R1", {
        "category": "bread", "event": "weekend", "status": "researched",
        "multiplier": 1.2, "confidence": "high", "reasoning": "...", "sources": [],
    })
    priors_store.approve("R1", "bread", "weekend")

    priors_store.save_researched("R1", {
        "category": "bread", "event": "weekend", "status": "researched",
        "multiplier": 3.0, "confidence": "low", "reasoning": "a different run", "sources": [],
    })
    assert priors_store.approved_multipliers("R1", "bread") == {"weekend": 1.2}


@requires_mongo
def test_priors_are_per_tenant(priors_store):
    priors_store.save_researched("TENANT_A", {
        "category": "bread", "event": "weekend", "status": "researched",
        "multiplier": 1.5, "confidence": "high", "reasoning": "...", "sources": [],
    })
    priors_store.approve("TENANT_A", "bread", "weekend")

    assert priors_store.approved_multipliers("TENANT_A", "bread") == {"weekend": 1.5}
    assert priors_store.approved_multipliers("TENANT_B", "bread") == {}
