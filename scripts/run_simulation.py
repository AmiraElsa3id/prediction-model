"""Business simulation: what would each production policy have cost?

Run:  .venv/bin/python -m scripts.run_simulation

This is the evaluation that matters. WAPE measures accuracy; a bakery owner is paid in
EGP. The two rank models differently, and deliberately so: the newsvendor model gives up
some accuracy to buy a better cost profile, so judging it on WAPE alone understates it.

Every policy is charged the same way:

    production = the policy's forecast
    waste      = max(production - demand, 0) x unit_cost x spoilage_severity
    lost sales = max(demand - production, 0) x margin

The baseline is the manager's ACTUAL production from the generated history, so the
comparison is against what the bakery does today, not a strawman.

ALL FIGURES ARE SIMULATED. They project what the model would have achieved on generated
data with assumed unit economics -- they are not measurements from a real bakery.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core.evaluation import make_folds
from app.core.features import build_features
from app.core.generate import generate
from app.core.items import BY_SKU
from app.models.forecaster import (
    CalendarDecomposed, LightGBMQuantile, MovingAverage, SeasonalNaive,
)

HORIZON = 1
N_FOLDS = 6
TEST_DAYS = 28


def cost_of(production: np.ndarray, demand: np.ndarray, skus: pd.Series) -> dict:
    """Charge a production plan against realised demand."""
    cost = np.zeros(len(production))
    waste_units = np.zeros(len(production))
    short_units = np.zeros(len(production))
    revenue = np.zeros(len(production))

    for sku in skus.unique():
        item = BY_SKU[sku]
        sel = (skus == sku).to_numpy()
        p, d = production[sel], demand[sel]
        over = np.maximum(p - d, 0)
        under = np.maximum(d - p, 0)
        sold = np.minimum(p, d)

        waste_units[sel] = over
        short_units[sel] = under
        revenue[sel] = sold * item.margin
        cost[sel] = (
            over * item.unit_cost * item.spoilage_severity   # value destroyed
            + under * item.shortage_cost                     # margin + goodwill lost
        )

    total_demand = float(demand.sum())
    return {
        "total_cost_egp": float(cost.sum()),
        "waste_units": float(waste_units.sum()),
        "short_units": float(short_units.sum()),
        "realised_margin_egp": float(revenue.sum()),
        # Fill rate is the honest service metric: what share of demand did we actually
        # serve? The day-level stockout flag below counts being two units short out of
        # fourteen hundred as a full "stockout", which badly overstates the problem.
        "fill_rate": 1.0 - float(short_units.sum()) / total_demand if total_demand else 1.0,
        "waste_rate": float(waste_units.sum()) / max(float(production.sum()), 1.0),
        "stockout_day_rate": float((short_units > 0).mean()),
    }


def main() -> None:
    print("=" * 84)
    print("SIMULATED RESULTS -- projected on generated data, not measured in a bakery")
    print("=" * 84)

    raw = generate()
    feats = build_features(raw, horizon=HORIZON)
    folds = make_folds(feats["date"], n_folds=N_FOLDS, test_days=TEST_DAYS, horizon=HORIZON)

    models = {
        "seasonal_naive": SeasonalNaive(),
        "moving_average": MovingAverage(),
        "lgbm_newsvendor": LightGBMQuantile(censoring="impute", name="lgbm_newsvendor"),
        "decomposed_median": CalendarDecomposed(quantile=0.5, name="decomposed_median"),
        "decomposed_newsvendor": CalendarDecomposed(name="decomposed_newsvendor"),
    }

    # Collect the evaluation window once so every policy is scored on identical rows.
    test_frames = []
    preds: dict[str, list[np.ndarray]] = {k: [] for k in models}

    for fold in folds:
        train = feats[feats["date"] <= fold.train_end]
        test = feats[(feats["date"] >= fold.test_start) & (feats["date"] <= fold.test_end)]
        test = test[test["true_demand"].notna() & (test["is_closed"] == 0)]
        if train.empty or test.empty:
            continue
        test_frames.append(test)
        for name, model in models.items():
            preds[name].append(np.asarray(model.fit_predict(train, test), dtype=float))

    evaluated = pd.concat(test_frames, ignore_index=True)
    demand = evaluated["true_demand"].to_numpy(dtype=float)
    skus = evaluated["sku"]
    days = evaluated["date"].nunique()

    rows = []

    # Baseline: what the manager actually produced on these very days.
    manager = cost_of(evaluated["production_qty"].to_numpy(dtype=float), demand, skus)
    manager["policy"] = "manager (today)"
    rows.append(manager)

    for name in models:
        p = np.concatenate(preds[name])
        r = cost_of(p, demand, skus)
        r["policy"] = name
        rows.append(r)

    table = pd.DataFrame(rows).set_index("policy")
    baseline_cost = table.loc["manager (today)", "total_cost_egp"]
    table["vs_manager_pct"] = (baseline_cost - table["total_cost_egp"]) / baseline_cost * 100
    table["cost_per_day_egp"] = table["total_cost_egp"] / days

    print(f"\nEvaluation window: {days} days x {skus.nunique()} items "
          f"({len(evaluated):,} item-days)\n")
    print(
        table[[
            "total_cost_egp", "cost_per_day_egp", "vs_manager_pct",
            "waste_rate", "fill_rate", "waste_units", "short_units",
        ]]
        .round(2)
        .sort_values("total_cost_egp")
        .to_string()
    )

    best = table["total_cost_egp"].idxmin()
    saving_per_day = (baseline_cost - table.loc[best, "total_cost_egp"]) / days
    waste_cut = (
        (manager["waste_units"] - table.loc[best, "waste_units"]) / manager["waste_units"] * 100
    )

    print(f"\n{'=' * 84}\nHEADLINE (SIMULATED)\n{'=' * 84}")
    print(f"Best policy         : {best}")
    print(f"Saving vs manager   : {saving_per_day:,.0f} EGP/day"
          f"  ~ {saving_per_day * 30:,.0f} EGP/month")
    print(f"Waste reduction     : {waste_cut:.1f}% fewer units wasted")
    print(f"Waste rate          : {manager['waste_rate']:.1%} -> "
          f"{table.loc[best, 'waste_rate']:.1%} of production")
    print(f"Fill rate (service) : {manager['fill_rate']:.1%} -> "
          f"{table.loc[best, 'fill_rate']:.1%} of demand served")

    # Does the newsvendor quantile actually earn its keep vs a plain median model?
    med, nv = table.loc["decomposed_median"], table.loc["decomposed_newsvendor"]
    delta = med["total_cost_egp"] - nv["total_cost_egp"]
    print(f"\nNewsvendor vs median: {delta:+,.0f} EGP over the window "
          f"({delta / days:+,.0f}/day)")
    print("  -- the quantile trades accuracy for cost; if this is negative, the plain"
          "\n     median model is better and we should ship that instead.")


if __name__ == "__main__":
    main()
