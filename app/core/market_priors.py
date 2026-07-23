"""Market priors for the rule-based cold-start forecaster.

The rule-based layer needs calendar sensitivities -- how much each item moves in Ramadan,
on weekends, at Eid -- before any sales data exists. These are educated estimates, not
measurements. An LLM with knowledge of Egyptian food culture produces reasonable ones,
and the real data corrects them over the first ~90 days (after which the trained model
takes over the item entirely).

Two layers, both in `data/market_priors.json`:
  * category defaults -- transferable across restaurants (all desserts spike in Ramadan)
  * item overrides    -- the standout items (konafa is THE Ramadan sweet)

NOT here: absolute daily levels (how many croissants THIS bakery sells). An LLM cannot
know that; it comes from onboarding and self-corrects from data. See `RuleBasedForecaster`.

`generate_priors_llm()` can (re)produce this file from a restaurant's real menu via a
free-tier LLM, validating the output and falling back to the shipped file / catalogue
defaults on any failure -- the same safe pattern as the marketing copy generator.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from app.core.items import BY_SKU, Item

PRIORS_PATH = Path(__file__).resolve().parents[2] / "data" / "market_priors.json"

# Prior-file event key -> the Item attribute used as the fallback default.
KEY_TO_ATTR = {
    "weekend": "weekend_mult",
    "ramadan": "ramadan_mult",
    "ramadan_late": "ramadan_late_boost",
    "eid": "eid_mult",
    "kahk_peak": "kahk_peak_mult",
    "school": "school_mult",
    "payday": "payday_mult",
    "sham_el_nessim": "sham_mult",
    "holiday": "holiday_mult",
}

# A multiplier outside this range is almost certainly an LLM hallucination, not a real
# effect -- reject it and keep the catalogue default for that key.
MIN_MULT, MAX_MULT = 0.05, 60.0

# Neutral priors: no calendar effect at all. The fallback for an unknown category.
NEUTRAL_PRIORS = {key: 1.0 for key in KEY_TO_ATTR}


def _item_defaults(item: Item) -> dict[str, float]:
    """Fallback priors straight from the item catalogue."""
    return {key: float(getattr(item, attr)) for key, attr in KEY_TO_ATTR.items()}


def _clean(values: dict) -> dict[str, float]:
    """Keep only known keys with sane numeric values."""
    out = {}
    for k, v in values.items():
        if k in KEY_TO_ATTR and isinstance(v, (int, float)) and MIN_MULT <= v <= MAX_MULT:
            out[k] = float(v)
    return out


def _read(path: Path | str = PRIORS_PATH) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def category_priors(category: str, path: Path | str = PRIORS_PATH) -> dict[str, float]:
    """Calendar sensitivities for a category, for products outside the built-in catalogue.

    Used by the RestoMind bridge: a real restaurant's products carry their own category,
    so we resolve priors by category name rather than by our fixed SKUs. Unknown category
    -> neutral (no calendar effect), which is the safe default.
    """
    cats = _read(path).get("categories", {})
    priors = dict(NEUTRAL_PRIORS)
    priors.update(_clean(cats.get(category, {})))
    return priors


def load_priors(path: Path | str = PRIORS_PATH) -> dict[str, dict[str, float]]:
    """Resolve final per-item priors: catalogue defaults <- category <- item override.

    Always returns a complete set for every known SKU, so a partial or missing file
    degrades gracefully rather than leaving gaps.
    """
    data: dict = {}
    p = Path(path)
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = {}

    categories = data.get("categories", {})
    overrides = data.get("item_overrides", {})

    resolved: dict[str, dict[str, float]] = {}
    for sku, item in BY_SKU.items():
        priors = _item_defaults(item)
        priors.update(_clean(categories.get(item.category, {})))
        priors.update(_clean(overrides.get(sku, {})))
        resolved[sku] = priors
    return resolved


# --------------------------------------------------------------------------------------
# Optional: (re)generate the priors file from a menu via a free-tier LLM.
# --------------------------------------------------------------------------------------


def _llm_prompt(items: list[Item]) -> str:
    menu = "\n".join(f"- {i.sku} ({i.name_ar}, {i.category})" for i in items)
    keys = ", ".join(KEY_TO_ATTR)
    return (
        "أنت خبير في سوق المخابز والمطاعم المصرية. المطلوب: قدّر حساسية الطلب لكل صنف "
        "للمناسبات المصرية كمُضاعِفات (1.0 = لا تغيير، 2.0 = الضعف، 0.5 = النصف).\n\n"
        f"المناسبات: {keys}\n"
        "(ramadan = نهار رمضان، ramadan_late = آخر 10 أيام، kahk_peak = ذروة موسم الكحك قبل عيد الفطر)\n\n"
        f"الأصناف:\n{menu}\n\n"
        "أرجِع JSON فقط بالشكل: "
        '{ "item_overrides": { "SKU": { "ramadan": 4.5, ... } } }\n'
        "اذكر فقط المناسبات المؤثرة لكل صنف. من غير أي كلام تاني."
    )


def generate_priors_llm(
    items: list[Item] | None = None,
    save_to: Path | str | None = None,
    api_key: str | None = None,
) -> tuple[dict, str]:
    """Ask an LLM to produce priors for a menu; validate and fall back on any failure.

    Returns (resolved_priors, source) where source is 'llm' or 'fallback'. With no API
    key configured this returns the shipped file / catalogue defaults and source
    'fallback' -- so the feature is always usable, and better with a key.
    """
    items = items or list(BY_SKU.values())
    api_key = api_key or os.getenv("LLM_API_KEY")

    if not api_key:
        return load_priors(), "fallback"

    try:
        import httpx

        base_url = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
        model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
        resp = httpx.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": _llm_prompt(items)}],
                "temperature": 0.4,
                "response_format": {"type": "json_object"},
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        payload = json.loads(resp.json()["choices"][0]["message"]["content"])

        # Validate: keep only clean overrides for known SKUs.
        overrides = {
            sku: cleaned
            for sku, vals in payload.get("item_overrides", {}).items()
            if sku in BY_SKU and (cleaned := _clean(vals))
        }
        if not overrides:
            return load_priors(), "fallback"

        merged = {"categories": {}, "item_overrides": overrides}
        if save_to:
            Path(save_to).write_text(
                json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        # Resolve against the just-produced overrides.
        base = load_priors()  # catalogue + shipped categories
        for sku, vals in overrides.items():
            base[sku].update(vals)
        return base, "llm"
    except Exception:
        # Network, rate limit, malformed JSON -- fall back rather than fail.
        return load_priors(), "fallback"


if __name__ == "__main__":
    priors, source = generate_priors_llm()
    print(f"priors source: {source}")
    for sku in ("SWEET_KONAFA", "PASTRY_CROISSANT", "SAVOURY_FETEER"):
        p = priors[sku]
        print(f"  {sku:18s} ramadan={p['ramadan']:.2f}  weekend={p['weekend']:.2f}  "
              f"sham={p['sham_el_nessim']:.2f}  eid={p['eid']:.2f}")
