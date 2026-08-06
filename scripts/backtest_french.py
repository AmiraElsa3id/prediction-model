"""
Walk-forward backtest of the CalendarDecomposed *architecture* against the real French
bakery Kaggle dataset, using France's own real calendar (`app/core/french_calendar.py`,
via the `holidays` library) -- the redo of this session's very first real-data test,
which went through the rule-based bridge with EGYPT's calendar and lost badly to naive
(WAPE 0.605). Same real data, same real dates, right calendar this time.

Run:  .venv/Scripts/python.exe scripts/backtest_french.py

Same reuse/adaptation boundary as backtest_m5.py / backtest_favorita.py -- see either
file's docstring for the full reasoning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core.features import add_lags
from app.core.evaluation import backtest, per_item_scores
from app.core.french_calendar import CALENDAR
from app.models.forecaster import SeasonalNaive, MovingAverage, LightGBMQuantile
from app.models.french_seasonality import FrenchCalendarEffects
from scripts.load_french import load_french_subset

HORIZONS = [1, 7]
N_FOLDS = 6
TEST_DAYS = 28


def _prep_base(df: pd.DataFrame) -> pd.DataFrame:
    cal = CALENDAR.feature_frame(df["date"].min(), df["date"].max())
    cal["date"] = pd.to_datetime(cal["date"])
    out = df.merge(cal, on="date", how="left")
    out["is_closed"] = 0
    out["is_outlier"] = 0
    out["sku_code"] = out["sku"].astype("category").cat.codes
    return out


def _quiet_clean_baseline(df: pd.DataFrame) -> pd.Series:
    quiet = (df["is_public_holiday"] == 0) & (df["is_august"] == 0)
    quiet_target = df["sales_qty"].where(quiet)
    clean = (
        quiet_target.groupby(df["sku"], sort=False)
        .transform(lambda s: s.shift(1).rolling(70, min_periods=5).mean())
    )
    clean = clean.groupby(df["sku"], sort=False).transform(lambda s: s.ffill().bfill())
    return clean.fillna(df["baseline_level"]).clip(lower=0.0)


class FrenchCalendarDecomposed:
    """demand = deseasonalised level (LightGBM) x calendar multiplier (Ridge, France's
    real holidays + August). Same two-stage architecture as the M5/Favorita variants."""

    name = "french_calendar_decomposed"

    def __init__(self, horizon: int = 1) -> None:
        self.horizon = horizon
        self._effects: FrenchCalendarEffects | None = None
        self._level: LightGBMQuantile | None = None
        self._train: pd.DataFrame | None = None

    def _deseasonalise(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["_mult"] = self._effects.multiplier(out)
        out["_deseason"] = out["sales_qty"] / out["_mult"]
        out = add_lags(out, horizon=self.horizon, target="_deseason")
        return out

    def fit(self, train: pd.DataFrame) -> "FrenchCalendarDecomposed":
        self._effects = FrenchCalendarEffects().fit(train)
        self._train = train

        deseason = self._deseasonalise(train)
        deseason["sales_qty"] = deseason["_deseason"]

        self._level = LightGBMQuantile(
            quantile=0.5, censoring="ignore", target_mode="ratio",
        ).fit(deseason)
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        combined = pd.concat([self._train, test], ignore_index=True)
        combined = combined.drop_duplicates(subset=["sku", "date"], keep="last")
        deseason = self._deseasonalise(combined)

        keys = pd.MultiIndex.from_frame(test[["sku", "date"]])
        indexed = deseason.set_index(["sku", "date"])
        rows = indexed.loc[keys].reset_index()

        level = self._level.predict(rows)
        mult = self._effects.multiplier(test)
        return np.clip(np.round(level * mult), 0, None)

    def fit_predict(self, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        return self.fit(train).predict(test)


def build_models(horizon: int) -> dict:
    return {
        "seasonal_naive": SeasonalNaive(),
        "moving_average": MovingAverage(),
        "lgbm_flat": LightGBMQuantile(quantile=0.5, censoring="ignore", target_mode="ratio"),
        "french_calendar_decomposed": FrenchCalendarDecomposed(horizon=horizon),
    }


def main() -> None:
    print("=" * 78)
    print("REAL FRENCH BAKERY DATA -- real sales, real calendar (France, `holidays` lib)")
    print("=" * 78)

    raw = load_french_subset()
    base = _prep_base(raw)

    all_rows = []
    for horizon in HORIZONS:
        feats = add_lags(base, horizon=horizon, target="sales_qty")
        feats["clean_baseline"] = _quiet_clean_baseline(feats)
        print(f"\n\n{'#' * 78}\n# HORIZON: {horizon} day(s) ahead\n{'#' * 78}")

        for name, model in build_models(horizon).items():
            results, summary = backtest(
                feats, fit_predict=model.fit_predict, target="sales_qty",
                n_folds=N_FOLDS, test_days=TEST_DAYS, horizon=horizon,
            )
            summary["model"] = name
            summary["horizon"] = horizon
            all_rows.append(summary)

            print(
                f"\n{name:28s} WAPE={summary['wape']:.4f}  MAE={summary['mae']:8.2f}  "
                f"RMSE={summary['rmse']:8.2f}  MASE={summary.get('mase', float('nan')):.3f}  "
                f"bias={summary['bias']:+8.2f}"
            )

            if name == "french_calendar_decomposed":
                worst = per_item_scores(results).head(5)
                print("   worst series by WAPE:")
                for sku, row in worst.iterrows():
                    print(f"     {sku:20s} WAPE={row['wape']:.3f}  bias={row['bias']:+8.2f}")

    table = pd.DataFrame(all_rows)
    print(f"\n\n{'=' * 78}\nSUMMARY\n{'=' * 78}")
    pivot = table.pivot(index="model", columns="horizon", values="wape")
    pivot.columns = [f"WAPE_h{h}" for h in pivot.columns]
    print(pivot.round(4).sort_values(pivot.columns[0]).to_string())

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
