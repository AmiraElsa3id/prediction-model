"""Compare all forecasters on a rolling-origin backtest.

Run:  .venv/bin/python -m scripts.run_backtest

Scores every model against BOTH targets:

  sales_qty    what a real deployment would see (censored by production)
  true_demand  ground truth, available only because the data is synthetic

The gap between the two is the censored-demand problem made visible. A model that looks
fine on sales and poor on true demand is quietly learning to under-produce.
"""

from __future__ import annotations

import pandas as pd

from app.core.features import build_features
from app.core.generate import generate
from app.core.evaluation import backtest, per_item_scores
from app.models.forecaster import (
    CalendarDecomposed, LightGBMQuantile, MovingAverage, SeasonalNaive,
)

HORIZONS = [1, 7]
N_FOLDS = 6
TEST_DAYS = 28


def build_models() -> dict:
    return {
        "seasonal_naive": SeasonalNaive(),
        "moving_average": MovingAverage(),
        # Censoring ablation, all at the median so the comparison is like-for-like.
        "lgbm_censor_ignore": LightGBMQuantile(
            quantile=0.5, censoring="ignore", name="lgbm_censor_ignore"
        ),
        "lgbm_censor_drop": LightGBMQuantile(
            quantile=0.5, censoring="drop", name="lgbm_censor_drop"
        ),
        "lgbm_censor_impute": LightGBMQuantile(
            quantile=0.5, censoring="impute", name="lgbm_censor_impute"
        ),
        # The production candidate: calendar decomposition + newsvendor quantile.
        "decomposed_median": CalendarDecomposed(quantile=0.5, name="decomposed_median"),
        "decomposed_newsvendor": CalendarDecomposed(name="decomposed_newsvendor"),
    }


def main() -> None:
    print("=" * 78)
    print("SIMULATED DATA -- generated, not measured from a real bakery")
    print("=" * 78)

    raw = generate()
    all_rows = []

    for horizon in HORIZONS:
        feats = build_features(raw, horizon=horizon)
        print(f"\n\n{'#' * 78}\n# HORIZON: {horizon} day(s) ahead\n{'#' * 78}")

        for name, model in build_models().items():
            results, summary = backtest(
                feats,
                fit_predict=model.fit_predict,
                target="sales_qty",
                eval_target="true_demand",
                n_folds=N_FOLDS,
                test_days=TEST_DAYS,
                horizon=horizon,
            )
            summary["model"] = name
            summary["horizon"] = horizon
            all_rows.append(summary)

            print(
                f"\n{name:24s} WAPE={summary['wape']:.4f}  MAE={summary['mae']:8.2f}  "
                f"RMSE={summary['rmse']:8.2f}  MASE={summary.get('mase', float('nan')):.3f}  "
                f"bias={summary['bias']:+8.2f}"
            )

            if name == "decomposed_newsvendor":
                worst = per_item_scores(results).head(4)
                print("   worst items by WAPE:")
                for sku, row in worst.iterrows():
                    print(f"     {sku:22s} WAPE={row['wape']:.3f}  bias={row['bias']:+8.2f}")

    table = pd.DataFrame(all_rows)
    print(f"\n\n{'=' * 78}\nSUMMARY (evaluated against TRUE demand)\n{'=' * 78}")
    pivot = table.pivot(index="model", columns="horizon", values="wape")
    pivot.columns = [f"WAPE_h{h}" for h in pivot.columns]
    print(pivot.round(4).sort_values(pivot.columns[0]).to_string())

    # The headline check: did the candidate beat the floor?
    for horizon in HORIZONS:
        sub = table[table["horizon"] == horizon].set_index("model")
        naive = sub.loc["seasonal_naive", "wape"]
        best = sub["wape"].idxmin()
        lift = (naive - sub.loc[best, "wape"]) / naive
        verdict = "BEATS" if best != "seasonal_naive" else "LOSES TO"
        print(
            f"\nh={horizon}: best is '{best}' -- {verdict} seasonal naive "
            f"({lift:+.1%} WAPE improvement)"
        )


if __name__ == "__main__":
    main()
