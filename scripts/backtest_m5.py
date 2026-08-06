"""
Walk-forward backtest of the CalendarDecomposed *architecture* against real M5 (Walmart)
data, using M5's own real calendar (`app/core/m5_calendar.py`) instead of Egypt's.

Run:  .venv/Scripts/python.exe scripts/backtest_m5.py

REUSED, unmodified:
  app.core.features.add_lags        -- lag/rolling engineering; gracefully no-ops the
                                        Egypt-only quiet-day branch when those columns
                                        are absent (checked explicitly in that function)
  app.core.evaluation.*             -- backtest(), make_folds(), wape(), mase(), score_all()
  app.models.forecaster.SeasonalNaive, MovingAverage, LightGBMQuantile

NOT reused (real coupling to Egypt found while wiring this up):
  app.models.forecaster.CalendarDecomposed  -- hardcodes `from app.models.seasonality
      import CalendarEffects`, so it cannot take a different calendar-effects estimator.
      M5CalendarDecomposed below is a small parallel orchestrator doing the identical
      two-stage fit (Ridge calendar decomposition -> LightGBM on deseasonalised ratio),
      just built on M5CalendarEffects instead.
  LightGBMQuantile's default `quantile=None` path looks up each item's newsvendor q*
      in Egypt's `BY_SKU` catalogue -- M5 skus are never in it, which would silently
      fit zero models. Worked around by always passing quantile=0.5 explicitly (this
      matches the already-existing `decomposed_median` variant compared in
      scripts/run_backtest.py -- not a new assumption introduced for this test).

SIMPLIFICATIONS vs the Egyptian pipeline (stated plainly, not hidden):
  M5 has no production/stockout-censoring signal the way POS data does, so the
  impute/drop censoring modes (which need `is_stockout`) don't apply -- `censoring=
  "ignore"` is used, itself one of the variants already compared in run_backtest.py.
  Outlier flagging (`flag_outliers`) is skipped rather than reused: its "explained"
  exemption checks Egypt-named calendar columns specifically, and reusing it unchanged
  would misclassify genuine M5 event spikes (Thanksgiving, etc.) as outliers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core.features import add_lags
from app.core.evaluation import backtest, per_item_scores
from app.core.m5_calendar import CALENDAR
from app.models.forecaster import SeasonalNaive, MovingAverage, LightGBMQuantile
from app.models.m5_seasonality import M5CalendarEffects
from scripts.load_m5 import load_m5_subset

HORIZONS = [1, 7]
N_FOLDS = 6
TEST_DAYS = 28


def _prep_base(df: pd.DataFrame) -> pd.DataFrame:
    """Attach M5's real calendar, plus the columns the reused Egyptian machinery
    unconditionally expects to find (is_closed/is_outlier: M5 has no such concept,
    so both are uniformly 0 -- stated in the module docstring, not hidden here)."""
    cal = CALENDAR.feature_frame(df["date"].min(), df["date"].max())
    cal["date"] = pd.to_datetime(cal["date"])
    out = df.merge(cal, on="date", how="left")
    out["is_closed"] = 0
    out["is_outlier"] = 0
    out["sku_code"] = out["sku"].astype("category").cat.codes
    return out


def _quiet_clean_baseline(df: pd.DataFrame) -> pd.Series:
    """M5's own version of `clean_baseline`: mean of recent QUIET days (no recorded
    event, no SNAP window) per sku, so the calendar Ridge fit's reference level isn't
    itself contaminated by the events it's trying to measure -- same reasoning as
    the Egyptian `clean_baseline`, different quiet-day definition."""
    quiet = (df["is_any_event"] == 0) & (df["snap_ca"] == 0)
    quiet_target = df["sales_qty"].where(quiet)
    clean = (
        quiet_target.groupby(df["sku"], sort=False)
        .transform(lambda s: s.shift(1).rolling(70, min_periods=5).mean())
    )
    clean = clean.groupby(df["sku"], sort=False).transform(lambda s: s.ffill().bfill())
    return clean.fillna(df["baseline_level"]).clip(lower=0.0)


class M5CalendarDecomposed:
    """demand = deseasonalised level (LightGBM) x calendar multiplier (Ridge, M5 events).

    Same two-stage architecture as `CalendarDecomposed`, wired to M5's real calendar
    instead of Egypt's -- see module docstring for exactly what's reused vs adapted.
    """

    name = "m5_calendar_decomposed"

    def __init__(self, horizon: int = 1) -> None:
        self.horizon = horizon
        self._effects: M5CalendarEffects | None = None
        self._level: LightGBMQuantile | None = None
        self._train: pd.DataFrame | None = None

    def _deseasonalise(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["_mult"] = self._effects.multiplier(out)
        out["_deseason"] = out["sales_qty"] / out["_mult"]
        out = add_lags(out, horizon=self.horizon, target="_deseason")
        return out

    def fit(self, train: pd.DataFrame) -> "M5CalendarDecomposed":
        self._effects = M5CalendarEffects().fit(train)
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
        "m5_calendar_decomposed": M5CalendarDecomposed(horizon=horizon),
    }


def main() -> None:
    print("=" * 78)
    print("REAL M5 (Walmart) DATA -- real sales, real calendar (calendar.csv)")
    print("=" * 78)

    raw = load_m5_subset()
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
                f"\n{name:24s} WAPE={summary['wape']:.4f}  MAE={summary['mae']:8.2f}  "
                f"RMSE={summary['rmse']:8.2f}  MASE={summary.get('mase', float('nan')):.3f}  "
                f"bias={summary['bias']:+8.2f}"
            )

            if name == "m5_calendar_decomposed":
                worst = per_item_scores(results).head(5)
                print("   worst series by WAPE:")
                for sku, row in worst.iterrows():
                    print(f"     {sku:32s} WAPE={row['wape']:.3f}  bias={row['bias']:+8.2f}")

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
