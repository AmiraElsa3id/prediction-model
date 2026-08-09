"""Egyptian Arabic promotional copy generation.

Two backends behind one interface:

  TemplateGenerator  hand-written Egyptian dialect templates. Free, instant, and
                     incapable of saying anything embarrassing.
  LLMGenerator       a free-tier hosted model, for copy that does not repeat.

The LLM always falls back to templates -- on API error, timeout, or failed validation.
A live demo must never stall on somebody's rate limit, and more importantly this text
is destined for a real bakery's public Facebook page. Unvalidated model output going
straight to a brand account is how you end up with a wrong price in public.

Validation is deliberately strict: the discount and item name must appear, the text
must actually be Arabic, and it must be short enough for a story card.
"""

from __future__ import annotations

import os
import random
import re
from dataclasses import dataclass
from typing import Mapping

from app.core.items import BY_SKU, Item

# Dialect note: these are Egyptian colloquial (masri), not Modern Standard Arabic.
# MSA reads as stiff and corporate in a neighbourhood bakery's social feed.
TEMPLATES = [
    "🔥 فرصة مميزة اليوم! احصل على {item} بخصم {discount}% حتى الساعة {close}. الكمية محدودة!",
    "طازج وبأعلى جودة 🥐 استمتع بـ {item} اليوم بخصم خاص {discount}%! اطلبه الآن قبل نفاد الكمية.",
    "عرض اليوم الخاص! 🎉 {item} بسعر {new_price} جنيه بدلاً من {old_price} جنيه. لا تفوت العرض!",
    "استمتع بخصم {discount}% على {item} 😍 العرض ساري حتى الساعة {close}. نتشرف بخدمتكم!",
    "تذوق الأفضل 🍰 احصل على {item} بخصم {discount}% اليوم. الأسبقية للأسبقية!",
    "عرض حصري لفترة محدودة ⏰ {item} بخصم {discount}% حتى الساعة {close}. اطلب الآن!",
]

# Refuse anything that over-promises or invents claims we cannot stand behind.
BANNED_PATTERNS = [
    r"مجان",        # free -- we are discounting, not giving away
    r"\b100\s*%",    # "100% off"
    r"مضمون",       # "guaranteed"
    r"أفضل في مصر",  # unverifiable superlative
]

MAX_LENGTH = 280
ARABIC_RE = re.compile(r"[؀-ۿ]")


@dataclass
class Offer:
    sku: str
    item_name_ar: str
    discount_pct: int
    old_price: float
    new_price: float
    copy_ar: str
    valid_until: str
    generator: str          # which backend actually produced the text
    hashtags: list[str]


def _fmt_price(price: float) -> str:
    """Format a price for display: one decimal for small non-integer prices, else whole.

    Without this, a 2.5 EGP item discounted becomes "2 بدل 2" (both rounded to int) --
    the offer looks meaningless. Cheap staples (baladi/fino bread) need the decimal.
    """
    p = round(float(price), 2)
    if p < 10 and p != int(p):
        return f"{p:.1f}".rstrip("0").rstrip(".")
    return str(int(round(p)))


def _validate(text: str, item: Item, discount_pct: int) -> tuple[bool, str]:
    """Gate every generated string, whoever produced it."""
    if not text or not text.strip():
        return False, "empty output"
    if len(text) > MAX_LENGTH:
        return False, f"too long ({len(text)} chars)"
    if len(ARABIC_RE.findall(text)) < 10:
        return False, "not Arabic"
    if item.name_ar not in text:
        return False, "item name missing"
    # The offer must be unambiguous, but there are two valid ways to say it: as a
    # percentage ("30% off") or as a price pair ("24 instead of 35"). Requiring the
    # percentage alone would reject perfectly good copy of the second kind.
    states_pct = str(discount_pct) in text
    states_prices = (
        _fmt_price(item.unit_price * (1 - discount_pct / 100)) in text
        and _fmt_price(item.unit_price) in text
    )
    if not (states_pct or states_prices):
        return False, "offer terms missing (neither discount % nor both prices)"
    for pattern in BANNED_PATTERNS:
        if re.search(pattern, text):
            return False, f"banned phrase matched {pattern!r}"
    return True, "ok"


class CopyGenerator:
    """Interface both backends implement."""

    name = "base"

    def generate(self, item: Item, discount_pct: int, close_time: str) -> str:
        raise NotImplementedError


class TemplateGenerator(CopyGenerator):
    """Deterministic Egyptian Arabic templates. Always available."""

    name = "template"

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def generate(self, item: Item, discount_pct: int, close_time: str) -> str:
        return self._rng.choice(TEMPLATES).format(
            item=item.name_ar,
            discount=discount_pct,
            close=close_time,
            new_price=_fmt_price(item.unit_price * (1 - discount_pct / 100)),
            old_price=_fmt_price(item.unit_price),
        )


class LLMGenerator(CopyGenerator):
    """Free-tier hosted LLM, OpenAI-compatible.

    Provider is configurable because the free tiers move around; Groq, Gemini and
    OpenRouter all expose an OpenAI-compatible endpoint, so switching is a base-URL
    change rather than a rewrite. With no API key set, this silently reports itself
    unavailable and the service uses templates.
    """

    name = "llm"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 8.0,
    ) -> None:
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.base_url = base_url or os.getenv(
            "LLM_BASE_URL", "https://api.groq.com/openai/v1"
        )
        self.model = model or os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _prompt(self, item: Item, discount_pct: int, close_time: str) -> str:
        new_price = round(item.unit_price * (1 - discount_pct / 100))
        return (
            "اكتب إعلان قصير لمخبز مصري بالعامية المصرية.\n"
            f"المنتج: {item.name_ar}\n"
            f"الخصم: {discount_pct}%\n"
            f"السعر القديم: {int(item.unit_price)} جنيه\n"
            f"السعر الجديد: {new_price} جنيه\n"
            f"العرض ساري لحد الساعة {close_time}\n\n"
            "القواعد:\n"
            f"- لازم تذكر اسم المنتج '{item.name_ar}' ونسبة الخصم '{discount_pct}%'\n"
            "- بالعامية المصرية مش الفصحى\n"
            "- أقل من 200 حرف، وممكن إيموجي\n"
            "- متقولش إن حاجة مجانية أو مضمونة\n"
            "- اكتب الإعلان بس من غير أي كلام تاني"
        )

    def generate(self, item: Item, discount_pct: int, close_time: str) -> str:
        if not self.available:
            raise RuntimeError("no LLM_API_KEY configured")

        import httpx

        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": self._prompt(item, discount_pct, close_time)}],
                "temperature": 0.9,
                "max_tokens": 200,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()


class OfferService:
    """Generates a validated offer, preferring the LLM and falling back to templates."""

    def __init__(self, llm: LLMGenerator | None = None, seed: int | None = None) -> None:
        self.llm = llm if llm is not None else LLMGenerator()
        self.templates = TemplateGenerator(seed=seed)

    def build_freeform(
        self, title_ar: str, price: float, discount_pct: int,
        close_time: str = "10 بالليل",
    ) -> dict:
        """Build offer copy for an arbitrary product (not a fixed SKU).

        Used by the RestoMind bridge, where products come from another system's menu.
        Same LLM+template+validation path as `build`, but keyed on a title and price
        rather than a catalogue item.
        """
        from types import SimpleNamespace

        item = SimpleNamespace(name_ar=title_ar, unit_price=price, category="")
        text, used = None, "template"
        if self.llm.available:
            try:
                candidate = self.llm.generate(item, discount_pct, close_time)
                if _validate(candidate, item, discount_pct)[0]:
                    text, used = candidate, "llm"
            except Exception:
                text = None
        if text is None:
            text = self.templates.generate(item, discount_pct, close_time)
            if not _validate(text, item, discount_pct)[0]:
                raise ValueError("offer copy failed validation")

        return {
            "item_name_ar": title_ar,
            "discount_pct": discount_pct,
            "old_price": round(price, 2),
            "new_price": round(price * (1 - discount_pct / 100), 2),
            "copy_ar": text,
            "generator": used,
        }

    def build(
        self, sku: str, discount_pct: int, close_time: str = "10 بالليل",
        valid_until: str | None = None,
        catalogue: Mapping[str, object] | None = None,
    ) -> Offer:
        if catalogue is not None:
            item = catalogue.get(sku)
            if item is None:
                raise KeyError(f"unknown SKU {sku!r}")
        else:
            item = BY_SKU[sku]
        text, used = None, "template"

        if self.llm.available:
            try:
                candidate = self.llm.generate(item, discount_pct, close_time)
                ok, _reason = _validate(candidate, item, discount_pct)
                if ok:
                    text, used = candidate, "llm"
            except Exception:
                # Any failure -- network, rate limit, malformed response -- falls
                # through to templates rather than failing the request.
                text = None

        if text is None:
            text = self.templates.generate(item, discount_pct, close_time)
            ok, reason = _validate(text, item, discount_pct)
            if not ok:  # templates are fixed, so this means a bad input
                raise ValueError(f"template output failed validation: {reason}")

        new_price = round(item.unit_price * (1 - discount_pct / 100), 2)
        return Offer(
            sku=sku,
            item_name_ar=item.name_ar,
            discount_pct=discount_pct,
            old_price=item.unit_price,
            new_price=new_price,
            copy_ar=text,
            valid_until=valid_until or close_time,
            generator=used,
            hashtags=["#مخبز", "#عروض", f"#{item.name_ar.replace(' ', '_')}", "#مصر"],
        )
