"""
Walk-forward backtest of the CalendarDecomposed *architecture* against real Favorita
(Corporación Favorita, Ecuador) data, using Favorita's own real, locale-filtered
calendar (`app/core/favorita_calendar.py`) -- third independent real-data test this
session, after the French bakery (rule-based bridge, failed) and M5 (this architecture,
beat naive by 27.6%).

Run:  .venv/Scripts/python.exe scripts/backtest_favorita.py

Same reuse/adaptation boundary as scripts/backtest_m5.py -- see that file's docstring
for the full reasoning (CalendarDecomposed hardcodes Egypt's CalendarEffects import,
LightGBMQuantile's default quantile=None couples to Egypt's BY_SKU, censoring="ignore"
since Favorita has no stockout signal, outlier flagging skipped for the same reason).

Favorita-specific: `sales_qty` here is already float (unit_sales can be fractional --
loose-weight bakery goods), and every (sku, date) that never appears in train.csv was
reindexed to an explicit 0 in load_favorita.py, not left missing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core.features import add_lags
from app.core.evaluation import backtest, per_item_scores
from app.core.favorita_calendar import CALENDAR
from app.models.forecaster import SeasonalNaive, MovingAverage, LightGBMQuantile
from app.models.favorita_seasonality import FavoritaCalendarEffects
from scripts.load_favorita import load_favorita_subset

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
    quiet = df["is_any_event"] == 0
    quiet_target = df["sales_qty"].where(quiet)
    clean = (
        quiet_target.groupby(df["sku"], sort=False)
        .transform(lambda s: s.shift(1).rolling(70, min_periods=5).mean())
    )
    clean = clean.groupby(df["sku"], sort=False).transform(lambda s: s.ffill().bfill())
    return clean.fillna(df["baseline_level"]).clip(lower=0.0)


class FavoritaCalendarDecomposed:
    """demand = deseasonalised level (LightGBM) x calendar multiplier (Ridge, Favorita
    events). Same two-stage architecture as CalendarDecomposed / M5CalendarDecomposed."""

    name = "favorita_calendar_decomposed"

    def __init__(self, horizon: int = 1) -> None:
        self.horizon = horizon
        self._effects: FavoritaCalendarEffects | None = None
        self._level: LightGBMQuantile | None = None
        self._train: pd.DataFrame | None = None

    def _deseasonalise(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["_mult"] = self._effects.multiplier(out)
        out["_deseason"] = out["sales_qty"] / out["_mult"]
        out = add_lags(out, horizon=self.horizon, target="_deseason")
        return out

    def fit(self, train: pd.DataFrame) -> "FavoritaCalendarDecomposed":
        self._effects = FavoritaCalendarEffects().fit(train)
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
        "favorita_calendar_decomposed": FavoritaCalendarDecomposed(horizon=horizon),
    }


def main() -> None:
    print("=" * 78)
    print("REAL FAVORITA (Ecuador) DATA -- real sales, real calendar (holidays_events.csv)")
    print("=" * 78)

    raw = load_favorita_subset()
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
                f"\n{name:30s} WAPE={summary['wape']:.4f}  MAE={summary['mae']:8.2f}  "
                f"RMSE={summary['rmse']:8.2f}  MASE={summary.get('mase', float('nan')):.3f}  "
                f"bias={summary['bias']:+8.2f}"
            )

            if name == "favorita_calendar_decomposed":
                worst = per_item_scores(results).head(5)
                print("   worst series by WAPE:")
                for sku, row in worst.iterrows():
                    print(f"     {sku:24s} WAPE={row['wape']:.3f}  bias={row['bias']:+8.2f}")

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
