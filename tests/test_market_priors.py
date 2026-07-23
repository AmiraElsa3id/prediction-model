"""Market priors: loading, graceful fallback, and effect on the rule-based forecast."""

import datetime as dt

from app.core.items import BY_SKU
from app.core.market_priors import generate_priors_llm, load_priors
from app.models.rule_based import RuleBasedForecaster


def test_load_priors_covers_every_item_with_all_keys():
    priors = load_priors()
    assert set(priors) == set(BY_SKU)
    required = {"weekend", "ramadan", "ramadan_late", "eid", "kahk_peak",
                "school", "payday", "sham_el_nessim", "holiday"}
    for p in priors.values():
        assert required <= set(p)


def test_missing_file_falls_back_to_catalogue_defaults():
    priors = load_priors(path="does_not_exist.json")
    # Still complete, using each item's own catalogue multipliers.
    assert priors["SWEET_KONAFA"]["ramadan"] == BY_SKU["SWEET_KONAFA"].ramadan_mult


def test_priors_reflect_the_analysis_file():
    priors = load_priors()
    # The standout facts the LLM analysis encodes.
    assert priors["SWEET_KONAFA"]["ramadan"] >= 3.0        # Ramadan sweet
    assert priors["SAVOURY_FETEER"]["sham_el_nessim"] >= 2.0  # Sham El-Nessim
    assert priors["PASTRY_CROISSANT"]["ramadan"] < 1.0     # breakfast falls in Ramadan


def test_custom_priors_change_the_forecast():
    d = dt.date(2025, 3, 15)  # Ramadan
    base = RuleBasedForecaster()
    tweaked_priors = {sku: dict(p) for sku, p in base.priors.items()}
    tweaked_priors["PASTRY_CROISSANT"]["ramadan"] = 0.2
    tweaked = RuleBasedForecaster(priors=tweaked_priors)
    assert tweaked.forecast("PASTRY_CROISSANT", d)["quantity"] < \
        base.forecast("PASTRY_CROISSANT", d)["quantity"]


def test_llm_generation_falls_back_without_api_key(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    priors, source = generate_priors_llm()
    assert source == "fallback"
    assert set(priors) == set(BY_SKU)
