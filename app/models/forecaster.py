"""Demand forecasting models.

Four models sharing one interface so the backtest can compare them fairly:

  SeasonalNaive      the floor every other model must clear
  MovingAverage      what the bakery manager already does, in effect
  LightGBMQuantile   one quantile booster per distinct newsvendor q*, all SKUs pooled
  CalendarDecomposed the production model -- level x calendar multiplier; see its own
                     docstring for why one booster over lags and calendar together does
                     not work

The naive baseline is not a formality. On short, noisy, per-item series it frequently
beats a tuned gradient booster, and shipping a model that loses to "same day last week"
would be worse than shipping nothing. If it wins here, it wins.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.core.features import feature_columns, training_mask
from app.core.items import BY_SKU


class BaseForecaster:
    """Common interface: fit on history, predict for a future frame."""

    name = "base"

    def fit(self, train: pd.DataFrame) -> "BaseForecaster":
        raise NotImplementedError

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def fit_predict(self, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        return self.fit(train).predict(test)


class SeasonalNaive(BaseForecaster):
    """Predict the same weekday from the previous week.

    Captures the weekly rhythm that dominates bakery demand, with zero parameters. It
    is blind to Ramadan and Eid, which is precisely where a calendar-aware model should
    pull ahead -- and where the demo's value story lives.
    """

    name = "seasonal_naive"

    def fit(self, train: pd.DataFrame) -> "SeasonalNaive":
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        pred = test["lag_same_dow"].to_numpy(dtype=float)
        # Fall back progressively when the lag is unavailable (new item, early history).
        for col in ("roll_mean_7", "roll_mean_28"):
            if col in test.columns:
                pred = np.where(np.isnan(pred), test[col].to_numpy(dtype=float), pred)
        return np.nan_to_num(pred, nan=0.0)


class MovingAverage(BaseForecaster):
    """The manager's own heuristic: trailing weekly mean plus a safety margin.

    Included so the business simulation can quantify improvement against what the
    bakery does today, rather than against a strawman.
    """

    name = "moving_average"

    def __init__(self, safety_margin: float = 1.25) -> None:
        self.safety_margin = safety_margin

    def fit(self, train: pd.DataFrame) -> "MovingAverage":
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        pred = test["roll_mean_7"].to_numpy(dtype=float) * self.safety_margin
        return np.nan_to_num(pred, nan=0.0)


@dataclass
class LightGBMQuantile(BaseForecaster):
    """Gradient-boosted quantile regression over calendar + lag features.

    One model across all SKUs rather than one per SKU. With ~11 items and two years of
    history, per-item models would each see too little data; pooling lets a rare event
    like Sham El-Nessim be learned from every item at once, with `sku_code` and the
    lag features carrying the item-specific level.

    The quantile objective is the point. `alpha` is set per item from its newsvendor
    q*, so low-markup same-day bread is forecast conservatively while long-shelf-life
    petit four is forecast generously. A single mean model cannot express that.
    """

    name: str = "lightgbm_quantile"
    quantile: float | None = None      # None => use each item's newsvendor q*
    n_estimators: int = 400
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_child_samples: int = 20
    censoring: str = "impute"          # "impute" | "drop" | "ignore"
    impute_quantile: float = 0.8       # tail level used to fill censored days
    target_mode: str = "ratio"         # "ratio" | "level"
    random_state: int = 42

    # Ratio targets are clipped to keep a near-zero baseline (a seasonal item waking up)
    # from producing an astronomically large training target.
    MIN_BASELINE: float = 1.0
    MAX_RATIO: float = 8.0

    _models: dict = field(default_factory=dict, init=False, repr=False)
    _features: list[str] = field(default_factory=list, init=False, repr=False)

    def _quantiles_to_fit(self, train: pd.DataFrame) -> list[float]:
        if self.quantile is not None:
            return [self.quantile]
        return sorted({BY_SKU[s].newsvendor_quantile for s in train["sku"].unique() if s in BY_SKU})

    def _impute_censored(
        self, train: pd.DataFrame, features: list[str],
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Recover demand on stockout days instead of discarding them.

        On a sell-out day the POS records `sales == production`, which is a *lower
        bound* on demand, not demand. Two obvious treatments are both wrong:

          ignore  -- trains on the capped value, teaching the model that the ceiling it
                     hit was the true demand. Under-production becomes self-reinforcing.
          drop    -- removes ~20% of rows, and specifically the busiest days. The
                     training set skews low and under-forecasting gets *worse*. Measured
                     on this data, dropping scored worse than ignoring.

        So we impute, in the spirit of EM for censored regression: fit once on the
        uncensored rows, predict what demand would have been on the censored ones, and
        take max(observed, predicted) -- demand was at least what we managed to sell.
        Then refit on everything.
        """
        import lightgbm as lgb

        base_mask = training_mask(train, exclude_stockouts=True)
        features = feature_columns(train, mode="level")
        censored = (
            (train["is_stockout"].fillna(0) == 1)
            & train["sales_qty"].notna()
            & train["lag_same_dow"].notna()
            & (train["is_closed"] == 0)
        )

        y = train["sales_qty"].to_numpy(dtype=float).copy()
        if not censored.any() or base_mask.sum() < 50:
            return train[base_mask | censored], y[(base_mask | censored).to_numpy()]

        # Impute from the upper tail, not the median. We already know demand exceeded
        # the cap that day -- that is what made it a stockout -- so the median of the
        # unconditional distribution is a known underestimate. Conditioning on
        # "demand > cap" means the right target is a high quantile.
        stage1 = lgb.LGBMRegressor(
            objective="quantile", alpha=self.impute_quantile,
            n_estimators=self.n_estimators, learning_rate=self.learning_rate,
            num_leaves=self.num_leaves, min_child_samples=self.min_child_samples,
            random_state=self.random_state, verbose=-1,
        )
        stage1.fit(train.loc[base_mask, features], train.loc[base_mask, "sales_qty"])

        imputed = stage1.predict(train.loc[censored, features])
        observed = train.loc[censored, "sales_qty"].to_numpy(dtype=float)
        # Demand was at least what we sold; trust the model only when it says "more".
        y[censored.to_numpy()] = np.maximum(observed, imputed)

        keep = (base_mask | censored).to_numpy()
        return train[keep], y[keep]

    def fit(self, train: pd.DataFrame) -> "LightGBMQuantile":
        import lightgbm as lgb

        self._features = feature_columns(train, mode=self.target_mode)

        if self.censoring == "impute":
            fit_df, y = self._impute_censored(train, self._features)
        else:
            mask = training_mask(train, exclude_stockouts=(self.censoring == "drop"))
            fit_df = train[mask]
            y = fit_df["sales_qty"].to_numpy(dtype=float)

        if fit_df.empty:
            raise ValueError("no trainable rows after masking")

        X = fit_df[self._features]

        if self.target_mode == "ratio":
            # Predict demand RELATIVE to the item's recent level, not the level itself.
            #
            # Trained on raw units, gradient boosting learns that `roll_mean_7` and
            # `lag_1` predict tomorrow better than any calendar flag -- true on the ~90%
            # of days that are ordinary. The model then tracks events only by following
            # sales downward *after* they start, and over-produces badly on the first
            # days of Ramadan, which is exactly when it matters most.
            #
            # Dividing out the level removes the shortcut. What is left to predict is
            # the multiplicative deviation, which is precisely what the calendar drives
            # -- and it matches how the demand actually composes.
            base = fit_df["baseline_level"].clip(lower=self.MIN_BASELINE).to_numpy()
            y = np.clip(y / base, 0.0, self.MAX_RATIO)

        # One booster per distinct quantile; items sharing a q* share a model.
        self._models = {}
        for q in self._quantiles_to_fit(train):
            model = lgb.LGBMRegressor(
                objective="quantile",
                alpha=q,
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                min_child_samples=self.min_child_samples,
                random_state=self.random_state,
                verbose=-1,
            )
            model.fit(X, y)
            self._models[q] = model
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        if not self._models:
            raise RuntimeError("call fit() before predict()")

        X = test[self._features]
        out = np.zeros(len(test), dtype=float)

        if self.quantile is not None:
            out = self._models[self.quantile].predict(X)
        else:
            # Route each row to the booster trained at its item's q*.
            q_by_row = test["sku"].map(
                lambda s: BY_SKU[s].newsvendor_quantile if s in BY_SKU else 0.5
            ).to_numpy()
            for q, model in self._models.items():
                sel = q_by_row == q
                if sel.any():
                    out[sel] = model.predict(X[sel])
            unresolved = ~np.isin(q_by_row, list(self._models))
            if unresolved.any():
                nearest = min(self._models)
                out[unresolved] = self._models[nearest].predict(X[unresolved])

        if self.target_mode == "ratio":
            # Convert the predicted deviation back into units.
            base = test["baseline_level"].clip(lower=self.MIN_BASELINE).to_numpy()
            out = np.clip(out, 0.0, self.MAX_RATIO) * base

        # Demand cannot be negative, and production is integral.
        return np.clip(np.round(out), 0, None)

    def feature_importance(self) -> pd.DataFrame:
        """Aggregate gain importance across the per-quantile boosters.

        Used by the recovery test and by the API's `contributing_factors`: if the
        Ramadan features are not near the top, the model has not learned the thing we
        built the whole calendar module for.
        """
        if not self._models:
            raise RuntimeError("call fit() before feature_importance()")
        total = np.zeros(len(self._features), dtype=float)
        for model in self._models.values():
            total += model.booster_.feature_importance(importance_type="gain")
        return (
            pd.DataFrame({"feature": self._features, "gain": total})
            .sort_values("gain", ascending=False)
            .reset_index(drop=True)
        )


@dataclass
class CalendarDecomposed(BaseForecaster):
    """demand = deseasonalised level (LightGBM) x calendar multiplier (ridge).

    The production model. Fitting one booster on calendar and lag features together does
    not work: the lags win, the calendar columns go unused, and the model reacts to
    Ramadan several days after it starts instead of anticipating it. See
    `app.models.seasonality` for the measurements behind that claim.

    Splitting the two jobs fixes it. The calendar multiplier is estimated from calendar
    features alone, so it cannot be crowded out; the booster then trains on
    deseasonalised demand, where its only job is tracking the slow-moving level. Because
    the multiplier is applied from the calendar rather than inferred from recent sales,
    it lands in full on the first day of Ramadan.
    """

    name: str = "calendar_decomposed"
    quantile: float | None = None
    censoring: str = "impute"
    horizon: int = 1
    n_estimators: int = 400
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_child_samples: int = 20
    random_state: int = 42

    _level: LightGBMQuantile | None = field(default=None, init=False, repr=False)
    _effects: object | None = field(default=None, init=False, repr=False)
    _train: pd.DataFrame | None = field(default=None, init=False, repr=False)

    def _deseasonalise(self, df: pd.DataFrame) -> pd.DataFrame:
        """Divide the calendar effect out and rebuild history features from the result.

        The lags must be recomputed, not merely rescaled: `lag_7` has to be divided by
        the multiplier from seven days earlier, not today's. Regenerating them from the
        deseasonalised series gets that right by construction.
        """
        from app.core.features import add_lags

        out = df.copy()
        out["_mult"] = self._effects.multiplier(out)
        out["_deseason"] = out["sales_qty"] / out["_mult"]
        out = add_lags(out, horizon=self.horizon, target="_deseason")
        return out

    def fit(self, train: pd.DataFrame) -> "CalendarDecomposed":
        from app.models.seasonality import CalendarEffects

        self._effects = CalendarEffects().fit(train)
        self._train = train

        deseason = self._deseasonalise(train)
        deseason["sales_qty"] = deseason["_deseason"]

        self._level = LightGBMQuantile(
            quantile=self.quantile, censoring=self.censoring, target_mode="ratio",
            n_estimators=self.n_estimators, learning_rate=self.learning_rate,
            num_leaves=self.num_leaves, min_child_samples=self.min_child_samples,
            random_state=self.random_state,
        ).fit(deseason)
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        if self._level is None or self._effects is None:
            raise RuntimeError("call fit() before predict()")

        # Deseasonalised lags for the test rows need the history behind them, so build
        # the combined frame and take back only the rows asked for.
        combined = pd.concat([self._train, test], ignore_index=True)
        combined = combined.drop_duplicates(subset=["sku", "date"], keep="last")
        deseason = self._deseasonalise(combined)

        keys = pd.MultiIndex.from_frame(test[["sku", "date"]])
        indexed = deseason.set_index(["sku", "date"])
        rows = indexed.loc[keys].reset_index()

        level = self._level.predict(rows)
        mult = self._effects.multiplier(test)
        return np.clip(np.round(level * mult), 0, None)

    def explain(self, row: pd.DataFrame) -> list[dict]:
        return self._effects.explain(row) if self._effects else []

    def calendar_multiplier(self, row: pd.DataFrame) -> np.ndarray:
        """The calendar component applied to these rows, for the seasonality endpoint."""
        if self._effects is None:
            raise RuntimeError("call fit() before calendar_multiplier()")
        return self._effects.multiplier(row)


MODELS: dict[str, type[BaseForecaster]] = {
    "seasonal_naive": SeasonalNaive,
    "moving_average": MovingAverage,
    "lightgbm_quantile": LightGBMQuantile,
    "calendar_decomposed": CalendarDecomposed,
}
