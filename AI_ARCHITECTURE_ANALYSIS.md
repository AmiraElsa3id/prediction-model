# AI Demand & Waste Platform for Egyptian Bakeries and Restaurants
## Architecture & Modelling Analysis (pre-implementation)

Date: 2026-07-20
Status: Analysis only. No code.

---

# 0. Executive summary — and three challenges to the brief

Before the phases, the four things that will most determine whether this project
succeeds or quietly fails:

### Challenge 1 — The three Kaggle datasets cannot train your production model.

Not "are weak." Cannot. Reasons developed in Phase 0, but in short: they are
almost certainly synthetic or semi-synthetic, they are not Egyptian, at least
two of them are not a `(store × product × day)` sales panel at all, and 100 days
of history cannot contain a Ramadan, an Eid, a summer, and a school year.

Their legitimate uses are: (a) scaffolding — build and unit-test the cleaning /
feature / validation pipelines against realistic-shaped data; (b) a demo for
stakeholders; (c) pretraining a cold-start prior. Not the model.

**Your actual dataset is the e-commerce platform's own order table.** The first
milestone of this project is not a model — it is instrumentation. See Phase 1.

### Challenge 2 — You are asking for the wrong prediction target.

The brief asks for "Predicted Quantity + Confidence Score." That implies a point
forecast of expected demand. Producing E[demand] is close to the worst possible
decision rule for a bakery, because the cost of one unit too many and one unit
too few are wildly different and asymmetric.

This is the classic **newsvendor problem**. The optimal production quantity is
not the mean, it is the **τ-th quantile of the demand distribution**, where:

```
τ = Cu / (Cu + Co)

Cu = cost of underproducing one unit = lost gross margin (+ goodwill)
Co = cost of overproducing one unit  = COGS lost to waste (− salvage value)
```

Worked example, baladi-style bread, sells 4 EGP, COGS 1.6 EGP:
- Cu = 2.4 EGP (margin forgone)
- Co = 1.6 EGP, but if unsold loaves are discounted at close for 2 EGP,
  salvage = 0.4 EGP above cost, so effective Co ≈ 1.2 EGP
- τ = 2.4 / (2.4 + 1.2) = **0.67** → produce the 67th percentile, i.e.
  deliberately above the mean.

Worked example, a 90 EGP gateau slice, COGS 35 EGP, zero salvage (discarded):
- Cu = 55, Co = 35 → τ = 0.61

Worked example, a fresh cream item with a 1-day shelf life and reputational
risk if served stale, COGS 40, and management judges a stockout cheap because
customers substitute:
- Cu = 15 (substitution recovers most of the margin), Co = 40 → τ = **0.27**
  → produce well *below* the mean.

Three products in the same shop, three completely different optimal quantiles.
A single point forecast cannot serve them. **Forecast the distribution; let a
separate, explicit, auditable policy layer convert distribution → quantity.**

This also directly answers Phase 4's question about overprediction sensitivity —
it isn't a modelling detail, it's the whole objective function.

### Challenge 3 — Sales are not demand, and ignoring this creates a death spiral.

Your order table records what was *sold*, which is `min(demand, stock)`. Every
sellout is a censored observation — true demand was higher by an unknown amount.

Train naively on sales and this happens: model under-forecasts → shop produces
less → sells out earlier → records even lower sales → model under-forecasts
further. Within a few retrain cycles the model has confidently talked a
profitable product down to nothing, and the dashboard shows *zero waste*, which
management will read as success.

Mitigation is mandatory and is a data-capture requirement, not a modelling
trick: **record the sellout timestamp for every product every day.** If an item
sold out at 15:40 and the shop closes at 22:00, that day's observation is
censored and must be either (a) excluded from the loss, (b) up-weighted via an
intraday sales-curve extrapolation, or (c) handled with a censored likelihood
(Tobit-style). Option (b) is the pragmatic MVP choice.

### Summary recommendation

| Question | Answer |
|---|---|
| Best MVP model | **LightGBM, global (pooled) model, quantile objective, multiple τ** |
| Best long-term model | Same, plus a hierarchical-reconciliation layer; TFT only if you exceed ~100 stores |
| Best online-learning model | None — nightly batch retrain. River only for drift monitors |
| Best architecture | **Modular monolith + async worker**, not 7 microservices |
| Best anomaly detection | Forecast-residual-based, not Isolation Forest on raw features |
| Time to useful MVP | ~10–12 weeks, of which ~4 are data capture, not ML |
| Realistic accuracy | High-volume staples: 12–20% MAPE. Long-tail items: 40–70% MAPE, and that is *normal* |

---

# Phase 0 — Dataset Validation

## 0.1 Method note (important)

The working directory is empty and I could not download or open these files.
Everything in §0.2–§0.4 is **prior belief, not observation**. Each dataset gets
an explicit confidence rating. Before a single line of modelling code, run the
profiling checklist in §0.6 and replace these hypotheses with facts.

## 0.2 Dataset 1 — "Food Wastage Data in Restaurant" (trevinhannibal)

**Believed schema** (confidence: medium): one row per *event* or per
*restaurant-meal-session*, with columns broadly like `Type of Food`,
`Number of Guests`, `Event Type`, `Quantity of Food`, `Storage Conditions`,
`Purchase History`, `Seasonality`, `Preparation Method`, `Geographical Location`,
`Pricing`, `Wastage Food Amount`.

**Synthetic?** Almost certainly. The tell is the column set: a mixture of
tidy low-cardinality categoricals with no join keys, no timestamps, no product
IDs, and a single tidy numeric target. Real operational waste data is messy,
timestamped, and keyed to a product catalogue.

**Verdict: REJECT for forecasting. Marginal use only.**

- No time index → cannot produce a time series. This alone disqualifies it from
  every feature in the brief that says "forecast."
- No product identity → cannot do multi-product.
- No store identity → cannot do multi-store.
- `Seasonality` as a *categorical column* rather than a derived date feature is
  a strong synthetic-data smell, and using it would leak: in production you do
  not have a hand-labelled "seasonality" field.
- The only defensible use: a toy regression to sanity-check the training
  harness end-to-end. Do not put it near the production model.

**Leakage risk: HIGH.** `Wastage Food Amount` and `Quantity of Food` are
mechanically related; any model will trivially learn `waste ≈ f(quantity)` and
report a flattering R². This is a fake result and will not survive contact with
real data.

## 0.3 Dataset 2 — "Restaurant Sales Report 2024-2025" (alexandchen)

**Believed schema** (confidence: low-medium): transaction or daily-aggregate
sales rows — plausibly `Order ID`, `Date`, `Item`, `Category`, `Quantity`,
`Price`, `Total`, `Payment Method`, maybe `Branch`.

**Verdict: CONDITIONALLY USEFUL — the only one of the three with real potential.**

It is the only candidate that could yield a `(product × day)` panel, which is
the minimum viable shape for this problem. Whether it does depends on facts I
cannot check: does it span ≥ 12 months? Is the item catalogue stable? Is there
more than one branch?

**Decision rule after profiling:**
- ≥ 12 months × ≥ 20 stable SKUs → use as the primary development dataset.
  Build the whole pipeline against it.
- < 12 months → use for pipeline development only; you cannot validate
  seasonality, so do not report seasonal accuracy from it.
- Uniform-looking daily quantities, suspiciously round prices, no weekday
  effect → it is synthetic; use for plumbing only and say so publicly in any
  stakeholder demo.

**Known limitations regardless:** not Egyptian (no Ramadan signal, no Friday
weekend, no payday effect), no waste/production columns (so it can train
*demand* but not *waste*), and no stock-level columns (so demand is censored
and you cannot even detect the censoring).

## 0.4 Dataset 3 — "Restaurant Inventory Management (100 Days)" (sujaldhanwani)

**Believed schema** (confidence: medium): daily per-ingredient inventory —
`Date`, `Ingredient`, `Opening Stock`, `Purchased`, `Consumed`, `Wastage`,
`Closing Stock`, perhaps `Shelf Life`.

**Verdict: REJECT as training data. USEFUL as a schema reference.**

- **100 days is fatal.** It cannot contain a full weekly-seasonality estimate
  with enough repeats, let alone monthly, let alone Ramadan. You need ≥ 2 full
  annual cycles to *learn* annual seasonality and ≥ 1 to even *see* it.
- It is **ingredient-level, not product-level.** Your platform sells products.
  Bridging ingredient ↔ product requires a bill-of-materials you do not have in
  this dataset. That mapping is a real and valuable thing to build — but it is
  a data-modelling project inside your e-commerce platform, not something this
  CSV provides.
- Its genuine value: it is a **good template for the inventory schema you
  should instrument in your own platform.** Steal the column design
  (opening / purchased / consumed / wasted / closing, with the identity
  `closing = opening + purchased − consumed − wasted` as a hard validation
  check), discard the rows.

## 0.5 Combined verdict

| Dataset | Real? | Time series? | Product-level? | Egypt? | Use |
|---|---|---|---|---|---|
| 1 — Food Wastage | Synthetic (likely) | No | No | No | Harness smoke-test only |
| 2 — Restaurant Sales | Unknown | Probably | Probably | No | **Primary dev dataset (if ≥12mo)** |
| 3 — Inventory 100d | Synthetic (likely) | Yes, but 100d | Ingredient-level | No | Schema template only |

**None of the three can validate the Egypt-specific features**, which are
precisely the features that will differentiate this product. Accept that: the
Egyptian calendar logic will ship as *encoded domain knowledge with
manager-tunable multipliers*, and only becomes statistically learned after
you have accumulated 2+ years of your own data.

## 0.6 Profiling checklist — run this first, for every dataset

1. Shape, dtypes, memory; head/tail/sample-20.
2. Per column: null count and null *pattern* (MCAR vs. structured — nulls
   clustered in time or in one store are a pipeline bug, not random noise).
3. Duplicates at three levels: exact row; business key `(store, product, date)`;
   near-duplicate (same key, differing values → a genuine integrity problem
   requiring a documented tie-break rule).
4. Cardinality of every categorical; flag any with > 100 levels for encoding
   strategy.
5. Date column: parse rate, min/max, **gaps** (missing days ≠ zero-demand days
   — a closed shop and a zero-sales day are different and must not be conflated),
   duplicated dates, future-dated rows, timezone consistency.
6. Target distribution: histogram, zero-inflation %, skew, kurtosis. If > 30%
   of `(product, day)` cells are zero, you have **intermittent demand** and
   standard regression metrics will mislead you (see Phase 4).
7. Per-series length distribution. Series with < 60 observations go to the
   cold-start path, not the main model.
8. Cross-column identities (e.g. `total == qty × price`; `closing == opening +
   purchased − consumed − wasted`). Violations reveal how the data was made.
9. Explicit leakage audit — see §0.8.

## 0.7 Outlier analysis — IQR vs Z-Score vs Isolation Forest vs LOF

### The framing in the brief is subtly wrong, and it matters.

The brief asks which method best *removes* outliers. But in demand data, **the
outliers are the product.** The day before Eid Al-Fitr, kahk demand goes up by
a multiple, not a percentage. That point is 10 standard deviations out. It is
also the single most commercially important observation in the entire dataset.
Delete it and you have built a model that is confidently, expensively wrong on
exactly the days that decide the year's profit.

So the correct question is not "which detector?" but **"which of my anomalies
are data errors, and which are real business events?"** — and no unsupervised
detector can answer that. Only the calendar can.

### The four methods, honestly compared

**IQR (`Q1 − 1.5·IQR`, `Q3 + 1.5·IQR`)**
- Univariate, non-parametric, robust to skew, trivially explainable to a
  non-technical bakery manager ("this is outside the normal range").
- Assumes a single population. Applied to a whole product's history it will
  flag every Friday and every holiday. **Only defensible when applied within a
  homogeneous group** — e.g. per `(product, day-of-week, non-holiday)` cell.
- Cost: O(n log n). Trivial.

**Z-Score (`|x − μ| / σ > 3`)**
- Assumes approximate normality. Demand data is right-skewed and often
  zero-inflated, so this assumption fails routinely.
- Worse: μ and σ are themselves corrupted by the outliers you are hunting
  (masking). A single Eid spike inflates σ enough to hide moderate anomalies.
- **Use the robust variant instead if you use it at all:** Modified Z-Score
  via median and MAD (`0.6745·(x − median)/MAD`), threshold 3.5. Same
  simplicity, no breakdown under contamination.
- Verdict: plain Z-score is the weakest of the four here. Reject.

**Isolation Forest**
- Multivariate, handles interactions (e.g. "high quantity is normal, and low
  temperature is normal, but high quantity *at* low temperature on a Tuesday
  is not"), scales near-linearly, no distance metric to tune.
- Downsides: stochastic (set `random_state`, or two runs disagree and you lose
  the manager's trust); `contamination` is an assumption you are forced to
  invent; and it gives no reason for a flag — bad for a system whose output is
  an alert a human must act on. Mitigate reason-opacity with SHAP on the
  anomaly score.
- **This is the right tool for the Surplus Detection feature (§Phase 5), not
  for data cleaning.**

**Local Outlier Factor**
- Detects *local* density anomalies — genuinely valuable when different
  products live at different demand scales, since a "normal" deviation for a
  1000-unit/day bread is enormous for a 5-unit/day specialty cake.
- Downsides: O(n²) in the naive form (a real problem at scale), `n_neighbors`
  is sensitive and unintuitive, degrades in high dimensions, and in its
  standard form it does not `predict()` on new data — you must use
  `novelty=True` and refit deliberately. That is an MLOps burden.
- Verdict: good idea, awkward in production. Skip for v1.

### Recommended strategy: a three-tier pipeline, calendar-first

```
TIER 1 — Hard validation rules (delete or fix; these are errors, not outliers)
  qty < 0, price ≤ 0, date in the future, qty > physical_capacity,
  identity violations (total ≠ qty × price)
  → route to a quarantine table; never silently drop.

TIER 2 — Calendar reconciliation (the key step)
  For every statistical anomaly, ask: does a known event explain it?
    Ramadan / Eid / public holiday / school break / marketing campaign
    / recorded local event / weather extreme
  → EXPLAINED   → keep, and ensure a feature encodes the cause.
                   This is signal. It is the most valuable data you own.
  → UNEXPLAINED → Tier 3.

TIER 3 — Statistical detection on the residual, within homogeneous groups
  Group by (store, product, day_of_week), exclude event days.
  Apply Modified Z-Score (median/MAD, threshold 3.5).
  Action: WINSORIZE at the 1st/99th percentile — do not delete.
    Deleting breaks the time index and corrupts every lag feature
    downstream. Winsorizing preserves the row and caps the leverage.
  Always write an `is_winsorized` flag column so the effect is auditable.
```

**Why winsorize rather than drop:** a time series with holes is not a time
series. `lag_7` silently becomes `lag_8` and every rolling window is wrong.
This is a very common and very quiet bug.

**Better still, where it applies:** don't fight outliers, change the loss.
Training on `log1p(qty)` and/or using a **Tweedie or Poisson objective**
(both native in LightGBM) matches count-like, zero-inflated, right-skewed
demand far better than squared error, and structurally de-emphasises extreme
values without discarding any data. This is often a bigger win than any amount
of outlier surgery.

## 0.8 Data leakage risks — the specific ones that will bite you

1. **Future information in lags.** The single most common failure. Predicting
   day T using a rolling mean whose window includes T. Every feature must be
   computable from data available at the moment the forecast is *made*, which
   for a "tomorrow's production" forecast is roughly 18:00 the previous
   evening — you may not even have today's full day of sales yet. Encode this
   as an explicit `available_at` timestamp per feature and assert it in tests.
2. **Waste ← production.** Waste is mechanically a function of production and
   sales. Any model predicting waste from production quantity will look
   brilliant and be useless. Predict *demand*; derive waste.
3. **Target encoding computed on the full dataset** before the time split.
   Category mean-encodings must be fit on the training fold only, expanding
   forward.
4. **Random train/test split.** Non-negotiable: never. See Phase 4.
5. **Price/promo known-in-advance confusion.** Promotions are legitimately
   known in advance (you plan them) — that is a valid future covariate.
   Realised discount rate is *not* known in advance. Keep these as separate,
   clearly-named columns and never let them mix.
6. **Scaler fit on all data.** Fit inside the CV fold.

## 0.9 Production readiness verdict

| Dataset | Production-ready |
|---|---|
| 1 | No — no time index, no keys, leakage-prone target |
| 2 | Not as-is; possibly adequate as a *development* proxy |
| 3 | No — 100 days, wrong granularity |

**None of them is production-ready. The production dataset does not exist yet;
you must create it.** That is Phase 1's main finding.

## 0.10 Pipeline designs

**Data Cleaning Pipeline** (idempotent, versioned, replayable):
```
raw ingest (immutable, append-only, partitioned by date)
  → schema conformance (Pydantic / pandera contract; fail loudly)
  → type coercion + timezone normalisation to Africa/Cairo
  → deduplicate on (store_id, product_id, date), documented tie-break
  → calendar spine join: reindex to a complete date range per series
      * distinguish CLOSED (null, excluded from loss)
        from OPEN-AND-SOLD-ZERO (genuine 0)
      * this distinction requires store opening-hours data — capture it
  → Tier 1 validation → quarantine table
  → Tier 2 calendar reconciliation
  → Tier 3 winsorize + flag
  → censoring flag from sellout timestamps
  → curated store (Parquet, partitioned; or a warehouse table)
```

**Feature Engineering Pipeline:** see Phase 2. Non-negotiable properties:
(a) one shared implementation used by both training and serving — never two
codepaths, that is how training/serving skew happens; (b) every feature
declares its `available_at` lead time; (c) point-in-time-correct joins.

**Data Validation Pipeline** (run on every batch, pre-train and pre-serve):
schema contract; null-rate thresholds per column; range checks; freshness
(`max(date)` within SLA); row-count vs. 7-day trailing median ±40%;
distribution drift vs. training reference (PSI / KS). Fail the pipeline, alert,
and **serve the previous model** rather than train on bad data.

**Data Quality Metrics** to dashboard: completeness %, freshness lag (hours),
duplicate rate, quarantine rate, schema-violation count, censored-observation
rate, series-coverage (% of active SKUs with ≥ 60 usable observations),
label-delay distribution.

---

# Phase 1 — Problem Analysis

## 1.1 Is the available data sufficient?

**No.** Clearly and specifically no.

For the three Kaggle datasets: no, for the reasons in Phase 0.

For your own platform: **partially, and the gaps are knowable.** Your order
table gives you sales, products, prices, stores, timestamps, and customers.
That is a genuinely strong foundation — arguably better than most restaurants
ever get. What it does *not* give you, and what you must add:

| Missing | Why it's essential | How to capture |
|---|---|---|
| **Production quantity per product per day** | Without it you can never compute waste, and "compare AI to manager quantity" is impossible | New table; manager entry at open |
| **Waste / unsold quantity at close** | The target for feature 3 and 4; the business KPI | End-of-day count. This is the hardest habit to establish — design the UX for 30 seconds of work |
| **Sellout timestamp** | Fixes demand censoring (Challenge 3) | Derivable if you track stock, else one tap |
| **Stock level / availability over the day** | Distinguishes "nobody wanted it" from "we ran out" | Inventory events |
| **Promotions & discounts with dates** | Large demand driver; a known-in-advance covariate | Likely already partly in the platform |
| **Store opening hours & closures** | Distinguishes closed-day from zero-demand day | Store config table |
| **Product shelf life & salvage value** | Required to compute the newsvendor τ per product | Product catalogue field |
| **Product BOM (ingredients)** | Ingredient-level purchasing, and the real cost model | Catalogue extension |
| **Weather (actual + forecast)** | Real effect on footfall, especially Cairo summer | External API |

**The single most important sentence in this document:** you cannot optimise
waste until you measure waste. If nothing else ships in month one, ship the
end-of-day waste capture screen.

## 1.2 Business questions the data CAN answer (once instrumented)

- How many units of product P should store S produce tomorrow? (core)
- How does demand shift across the week, and by how much on Friday?
- What is the Ramadan / Eid multiplier per category? (after 2 cycles)
- Which SKUs are structurally over-produced?
- Is a manager's entered quantity unusual given history and context?
- Which items currently on the shelf are likely unsold by close?
- What is the EGP value of avoidable waste, per store, per week?
- Which products should be discounted at 20:00, and by how much?

## 1.3 Business questions the data CANNOT answer

Be honest with stakeholders about these, early:

- **True unconstrained demand** — you only observe censored sales.
- **Causality of promotions.** Observational data confounds promo with the
  reason the promo was run. Only an A/B or switchback test gives you causal
  uplift. Do not let anyone report observational promo uplift as causal.
- **Cannibalisation and substitution.** When croissants sell out, do customers
  buy pain-au-chocolat or leave? This needs a cross-product demand model or
  designed experiments. Out of scope for v1 — but flag it, because a per-product
  independent model implicitly and wrongly assumes zero substitution.
- **Walk-in / non-platform demand** — invisible if some sales bypass the system.
- **New-product demand before launch** — cold start is an *educated prior*, not
  a forecast, and must be labelled as such in the UI.
- **Competitor actions, local construction, road closures** — unmodelled.
- **Why a specific day was anomalous**, without manual annotation. Build the
  annotation UI: let managers tag a day ("wedding order", "street closed",
  "power cut"). This is cheap and turns noise into features.

## 1.4 Assumptions being made — and which are fragile

| Assumption | Fragility | If violated |
|---|---|---|
| Sales ≈ demand | **Very fragile** | Doom spiral (Challenge 3) |
| Past patterns predict future | Moderate | Egypt's inflation/FX volatility shifts price sensitivity fast; retrain often |
| Product catalogue is stable | Fragile in bakeries — recipes and names change constantly | Series break silently. Need stable `product_id` decoupled from name |
| Managers will enter waste data | **Very fragile** — this is organisational, not technical | Whole waste feature dies. Mitigate with UX + a compliance metric per store |
| Stores behave similarly enough to pool | Moderate | Global model degrades; needs store embeddings + per-store fallback |
| Gregorian seasonality | **Violated by design** — Ramadan is lunar | Must handle explicitly (Phase 2) |
| Prices are stable | Violated in Egypt | Include real price and relative-price features |

## 1.5 Risks

**Data risk:** waste capture compliance decays after week 3. Track it as a
first-class metric; a store below 70% compliance gets its waste model disabled
rather than silently mispredicting.

**Model risk:** long-tail SKUs with 2 sales/day are near-unforecastable. Do not
promise accuracy on them. Segment the catalogue by volume and report separately.

**Business risk:** a manager follows a bad recommendation, has a stockout on a
big day, and loses trust permanently. Trust is asymmetric — one bad Eid costs
more than fifty good Tuesdays. Mitigations: always show a range not a number;
always show the reason; never auto-execute in year one; be conservative
(higher τ) around high-stakes days.

**Ethical/operational risk:** a system that optimises purely for waste
reduction will push toward chronic under-production, degrading customer
experience while showing green dashboards. **The primary KPI must be
profit (or margin net of waste cost), never waste alone.**

**Egypt-specific risk:** inflation and FX shifts change both costs and consumer
behaviour within months. Any model trained on 2-year-old price elasticity is
stale. Prefer relative-price features over absolute EGP.

## 1.6 Production challenges to expect

- Cold start: at launch you have *no* history, for everything. v1 must be
  useful on day one via heuristics, not day 400 via ML.
- Training/serving skew from duplicate feature code. Solve architecturally.
- Label delay: today's waste label arrives tomorrow morning.
- Ramadan shifts ~11 days earlier each Gregorian year; any date-based
  hard-coding will break annually.
- Connectivity: Egyptian branches may have unreliable internet. Recommendations
  must be cacheable and viewable offline.
- Arabic/English product names, RTL UI, mixed-script data entry — normalise
  early (Unicode NFKC, Arabic-Indic digit conversion, alef/hamza folding).
- Manager override behaviour is itself data. Log every override and its
  delta — it is your best signal of where the model is wrong.

---

# Phase 2 — Feature Engineering

All features below assume a target of `qty_sold(store, product, date)` on a
complete calendar spine, with a stated forecast origin (default: 18:00 the
previous day) that governs what is legitimately available.

## 2.1 Calendar features

| Feature | Why it matters | Generation | Expected impact |
|---|---|---|---|
| `day_of_week` | Dominant cycle in food retail | From date | **Very high** |
| `is_weekend` (**Fri–Sat in Egypt**) | Egypt's weekend is Friday–Saturday, not Sat–Sun. Getting this wrong invalidates the strongest feature you have | From date | **Very high** |
| `is_friday` | Distinct beyond "weekend": Jumu'ah prayer creates a sharp midday dip then a large post-prayer family surge | From date | **Very high** |
| `week_of_month`, `day_of_month` | Payday effect (below) | From date | High |
| `month`, `quarter` | Annual seasonality | From date | Medium |
| `day_of_year` sin/cos | Smooth annual cycle without 365 dummies | Fourier terms, 2–4 harmonics | Medium |
| `is_school_period` | Egyptian school year ≈ mid-Sept → early June, mid-year break ≈ late Jan. Drives morning pastry and sandwich demand hard | Calendar table | High for bakeries |
| `is_summer` / `temp_band` | Cairo summer suppresses hot food, lifts cold drinks and ice cream | From date + weather | High for mixed menus |

**Encode day_of_week as a categorical for tree models, never as an integer** —
integer encoding tells the model Sunday(0) and Saturday(6) are maximally
distant, when they are adjacent.

## 2.2 Egypt-specific features (your differentiator — build these carefully)

| Feature | Why | How |
|---|---|---|
| `is_ramadan` | Total regime change: daytime food sales collapse, iftar (sunset) demand explodes, suhoor creates a second late-night peak. Not a multiplier — a different distribution shape | Hijri calendar library (`hijri-converter` / `ummalqura`), **plus a manually confirmed table**, because Egypt's Dar al-Ifta announcement can shift the start by a day. Never compute it purely algorithmically |
| `ramadan_day_index` (1–30) | Demand is not flat across Ramadan: strong in week 1, dips mid-month, rises again before Eid | Derived |
| `days_to_iftar` / intraday | If you ever forecast intraday, iftar time (sunset) moves daily and is the single organising fact of the day | Sunset table for the store's city |
| `is_eid_al_fitr` + `days_to_eid` (−7..+3) | **Kahk season.** The pre-Eid week is the highest-revenue week of the year for Egyptian bakeries, often by a wide margin. Under-forecasting here is the most expensive mistake in the whole system | Hijri + confirmed table |
| `is_eid_al_adha` + `days_to_eid` | Meat-centric; suppresses bakery, boosts some restaurant categories; multi-day closures common | Hijri |
| `is_coptic_fast` (Great Lent ~55d pre-Coptic-Easter; Nativity Fast; Apostles' Fast) | ~10–15% of Egyptians observe; drives a large, predictable swing toward vegan/plant items. Almost no commercial system models this — genuine edge | Coptic calendar table |
| `is_sham_el_nessim` | Monday after Coptic Easter; national picnic day; huge feseekh/outdoor-food and bakery spike | Movable-feast table |
| `is_public_holiday` | Jan 7 (Coptic Christmas), Jan 25, Apr 25, May 1, Jun 30, Jul 23, Oct 6 + movable Islamic dates | Curated table, reviewed annually |
| `is_payday_window` (`day_of_month ∈ {28..31, 1..5}`) | Egyptian salaries are predominantly month-end. Spending is strongly front-loaded after payday and tapers hard by day 20–25 | From date |
| `days_since_payday` | Smoother version of the above; usually outperforms the binary flag | From date |

**Design note:** ship these as a **versioned `egypt_calendar` reference table**,
not scattered `if` statements. One table, one owner, reviewed each year,
with a test that asserts the next 24 months are populated. This table is a
genuine asset of the business.

**Critical honesty on Ramadan/Eid:** with < 2 years of your own data, the model
**cannot learn** these effects — it will have seen one or zero examples. For
v1, apply **explicit, manager-visible multipliers** as a post-model adjustment
layer (e.g. "kahk: ×6 on Eid−3"), seeded from domain interviews and adjusted by
the manager. Show them as a separate line in the UI: "base forecast 120,
Eid adjustment ×4.5 → 540." Transparent, correctable, and it accumulates the
labelled data that lets the model learn it properly by year three.

## 2.3 Time-series features

Computed strictly with data available at forecast origin (i.e. lag ≥ 1, and
ideally ≥ 2 if today's sales aren't final at 18:00).

| Feature | Why | Notes |
|---|---|---|
| `lag_1, lag_2, lag_3` | Short-term level & momentum | Most predictive single group |
| `lag_7, lag_14, lag_21, lag_28` | Same-weekday history — matches the dominant cycle | **Very high value** |
| `lag_364` | Same weekday one year prior (364 = 52×7 preserves weekday alignment; `lag_365` does not) | Needs 2 yrs data |
| `rolling_mean_7/14/28` | Level | Shift by lag first |
| `rolling_median_7/28` | Robust level, resistant to one spike | Prefer over mean for low-volume SKUs |
| `rolling_std_7/28` | **Volatility → directly feeds safety stock and the confidence interval** | High value |
| `ewm_alpha_{0.3,0.1}` | Recency-weighted level; adapts faster than a flat window | Good for trending items |
| `rolling_mean_same_dow_4` | Mean of last 4 same weekdays | Often the strongest single feature |
| `zero_rate_28` | Intermittency indicator; routes to a different model path | Important for long tail |
| `days_since_last_sale` | Intermittent-demand signal (Croston-like) | For long tail |
| `trend_ratio` = `rmean_7 / rmean_28` | Is this item accelerating or decaying? | High value |
| `stockout_rate_28` | Flags a censored, under-observed series | Prevents doom spiral |

**Implementation warning:** all rolling features must be computed
`groupby(store, product)` on a *complete, gap-free* date index, then shifted.
Computing rollings on a gappy frame is the most common bug in this whole class
of system and it fails silently — it produces good backtests and bad production.

## 2.4 Business & product features

| Feature | Why | Notes |
|---|---|---|
| `product_category` / `subcategory` | Enables pooling and cold start | Target-encode within fold |
| `price`, `price_change_pct` | Elasticity; critical under Egyptian inflation | |
| `relative_price` = price / category median | More stable than absolute EGP across inflation | **Prefer this** |
| `is_promo`, `discount_pct`, `promo_type` | Large uplift driver; known in advance | Keep planned vs. realised separate |
| `days_since_promo`, `days_to_promo` | Pull-forward and post-promo trough are real | |
| `store_id`, `store_type`, `city`, `district` | Location drives level & pattern | Categorical, not one-hot, for LightGBM |
| `store_footfall_lag` | Separates "fewer customers" from "same customers buying less" | If available |
| `product_age_days` | New products have launch curves | |
| `shelf_life_days`, `salvage_value` | **Feeds the newsvendor τ, not the model** | Policy layer input |
| `is_new_product` | Routes to cold-start path | |

## 2.5 Operational features

`opening_stock`, `production_qty`, `remaining_stock_at_close`, `waste_qty`,
`sellout_time`, `hours_available`.

**Handle with care — these are the highest leakage risk in the system.** For a
*demand* forecast made the previous evening, today's production and waste are
not available. Use them only as **lagged** features (`waste_lag_1`,
`prod_lag_1`, `sellout_time_lag_1`) and as targets for the waste/surplus models.
`sellout_time_lag_*` is genuinely valuable: repeated early sellouts are direct
evidence of suppressed demand.

## 2.6 Environmental features

| Feature | Why | Caveat |
|---|---|---|
| `temp_max`, `temp_min` | Hot-food vs. cold-item split; Cairo 40°C days suppress dine-in | **Use the forecast, not the actual** — at prediction time only the forecast exists. Train on forecasts too, or you create train/serve skew |
| `is_rain`, `precip_mm` | Rain suppresses footfall; matters in Alexandria and Delta more than Cairo | Rare in Upper Egypt; may be near-constant → low value there |
| `is_khamaseen` / high wind-dust | Egyptian sandstorm season (roughly March–May) genuinely suppresses street footfall | Derive from wind + visibility |
| `humidity` | Affects both comfort and dough proofing | Low priority |

**Cost/benefit:** weather adds a live external dependency, an API cost, and a
failure mode. It typically yields a few % accuracy. **Defer to v2** and design
the feature pipeline so it can be added without refactoring — but ship v1
without it.

## 2.7 Recommended v1 feature set (deliberately small)

Do not build all ~80 features for the MVP. Start with ~25 and add on evidence:

`dow`(cat), `is_friday`, `is_weekend`, `day_of_month`, `days_since_payday`,
`month`, `is_ramadan`, `ramadan_day_index`, `days_to_eid_fitr`,
`days_to_eid_adha`, `is_public_holiday`, `is_school_period`,
`lag_1,2,7,14,28`, `rmean_7,28`, `rmedian_7`, `rstd_7`, `rmean_same_dow_4`,
`trend_ratio`, `zero_rate_28`, `stockout_rate_28`, `relative_price`,
`is_promo`, `discount_pct`, `product_category`(cat), `store_id`(cat),
`product_age_days`.

Then run permutation importance and prune. Fewer features = faster retrains,
fewer breakages, easier explanation to the manager.

---

# Phase 3 — Model Comparison

Evaluation context: sparse-to-moderate volume, strong multi-seasonality,
hundreds-to-thousands of `(store, product)` series, many of them short,
heavy exogenous-calendar dependence, and a hard requirement for explainability
to non-technical users.

## 3.1 The framing decision that matters more than the model choice

**Local models (one model per series) vs. Global models (one model, all series).**

Prophet, ARIMA, SARIMA are *local*: they fit each `(store, product)` series
independently. With 30 stores × 200 products that is **6,000 models** to train,
version, monitor, and retrain nightly. Each sees only its own short history and
cannot learn from any other. New products get nothing.

Gradient-boosted trees used properly are *global*: one model over all series,
with `store_id` and `product_id` as features. It learns "Fridays are +40% for
pastries" from all 6,000 series jointly. It handles new products the moment they
have a category. One artifact to deploy.

**For this problem, global beats local decisively.** This single architectural
choice matters more than which of LightGBM/XGBoost/CatBoost you pick. It is
also why the "classical time-series" models score poorly below — not because
they are bad models, but because they are the wrong *shape* for a multi-product,
multi-store platform.

## 3.2 Model-by-model

### Prophet
- **Pros:** minimal tuning; handles holidays via an explicit regressor list
  (Ramadan/Eid fit its API cleanly); decomposable and visually explainable to a
  business audience; robust to missing data; native uncertainty intervals.
- **Cons:** local-only (one model per series); slow at 6,000 series;
  fundamentally a curve-fitter designed for smooth, high-volume, business-metric
  series — it performs **poorly on low-count, intermittent, integer demand**,
  which is most of a bakery catalogue; can predict negatives without a `floor`;
  weak with many exogenous features; no incremental learning; Meta's
  maintenance investment has visibly declined.
- **Data need:** ≥ 2 seasonal cycles per series.
- **Compute:** medium per series, prohibitive in aggregate.
- **Interpretability:** high. **Scalability:** poor. **Online:** none.
- **FastAPI:** awkward — model objects are large and slow to load; 6,000 of
  them is an artifact-management problem.
- **Expected on your data:** mediocre. Reasonable for your top 10 high-volume
  SKUs, poor for the long tail. **Verdict: not the primary. Good baseline.**

### River
- **Pros:** genuine online/incremental learning; tiny memory; adapts instantly
  to regime change; excellent built-in drift detectors (ADWIN, Page-Hinkley);
  very fast single-instance predict.
- **Cons:** its models are consistently weaker than batch GBDT on tabular
  problems; small ecosystem; hard to backtest and reason about ("what did the
  model know when?"); a continuously-mutating model is a governance and
  reproducibility headache; risks learning from a bad week.
- **Verdict: reject as the forecasting model. Adopt for drift detection.**
  This is the honest answer to "which model supports online learning" — River
  does, and you still should not use it for the forecast. **You do not need
  online learning here.** Demand does not change between 02:00 and 06:00. A
  nightly batch retrain (or even weekly) captures essentially all adaptivity
  that matters, with full reproducibility. Online learning is solving a problem
  you don't have.

### XGBoost
- **Pros:** excellent accuracy; mature and battle-tested; huge ecosystem;
  quantile ("pseudo-Huber"/`reg:quantileerror`) support in recent versions;
  strong GPU support; SHAP integration.
- **Cons:** historically weaker native categorical handling than LightGBM/
  CatBoost (improved but still less ergonomic); slower training than LightGBM
  on wide tabular data; no extrapolation beyond the training range of the
  target — a real limitation for a growing business (mitigate by modelling
  ratios/log-differences rather than raw levels).
- **Interpretability:** good (SHAP). **Scalability:** very good.
- **FastAPI:** easy — single small artifact, millisecond inference.
- **Verdict: a strong choice, narrowly second to LightGBM.**

### LightGBM — **RECOMMENDED**
- **Pros:** fastest training of the three (leaf-wise growth + histogram
  binning) — matters a lot when you retrain nightly and run walk-forward CV
  with dozens of folds; **native categorical support** (perfect for
  `store_id`, `product_id`, `category`, `dow`); **native quantile objective**,
  which is exactly what the newsvendor framing requires; also native Poisson
  and Tweedie objectives, which fit count-like zero-inflated demand far better
  than L2; small artifacts; excellent SHAP support; trivially deployable.
- **Cons:** can overfit small datasets without careful `num_leaves` /
  `min_data_in_leaf` tuning; leaf-wise growth is more overfit-prone than
  level-wise; no target extrapolation; requires manual feature engineering for
  time structure (this is a real cost, but the features in Phase 2 *are* the
  domain knowledge, so it is cost you should want to pay).
- **Data need:** works from ~1,000 rows total; shines from ~50k.
- **Compute:** low. Thousands of series, dozens of features, minutes on a
  single CPU box. No GPU needed.
- **Interpretability:** good — SHAP gives per-prediction reasons you can
  surface directly in the manager's UI ("+40% because Friday, +15% because
  payday week").
- **Scalability / real-time / FastAPI:** excellent. One artifact, sub-10ms
  predictions, easy to containerise.
- **Incremental learning:** partial (`init_model` warm start) — but prefer full
  nightly retrain for reproducibility.
- **Verdict: best MVP model AND best production model.**

### CatBoost
- **Pros:** best-in-class categorical handling via ordered target statistics —
  genuinely strong when you have high-cardinality IDs like `product_id`;
  ordered boosting reduces target leakage; excellent out-of-box defaults
  (least tuning of the three); good uncertainty support.
- **Cons:** slower training and larger artifacts than LightGBM; heavier
  dependency; smaller community.
- **Verdict: a legitimate alternative to LightGBM, especially if `product_id`
  cardinality is very high.** Benchmark it; the gap will be small. Choose on
  measured accuracy, and break ties on retrain speed → LightGBM.

### Random Forest
- **Pros:** almost no tuning; robust; hard to overfit badly; parallel;
  quantile variant available (`quantile-forest`).
- **Cons:** meaningfully less accurate than boosting on structured tabular
  data; large memory footprint; cannot extrapolate; averaging smooths away
  exactly the sharp peaks (Eid) you most need.
- **Verdict: use as the sanity-check baseline, not the product.**

### ARIMA
- **Pros:** well-understood theory; good on short, stationary, univariate
  series; cheap; strong statistical uncertainty quantification.
- **Cons:** no seasonality (that's SARIMA); local-only; requires stationarity
  work; poor with exogenous regressors (SARIMAX helps, awkwardly); very poor on
  intermittent/count data; 6,000 models.
- **Verdict: reject.**

### SARIMA / SARIMAX
- **Pros:** handles one seasonal period properly; exogenous regressors via
  SARIMAX; principled intervals.
- **Cons:** **only one seasonal period** — you have weekly *and* annual *and*
  lunar. This is disqualifying; SARIMA cannot represent Ramadan at all without
  heavy exogenous hacking. Slow to fit, local-only, order selection is fiddly
  at scale.
- **Verdict: reject.** Useful only as a single-series academic baseline.

### LSTM
- **Pros:** captures nonlinearity and long dependencies; can be global
  (multi-series with embeddings); handles multivariate input.
- **Cons:** needs far more data than you have (rule of thumb: thousands of
  observations per series, or a very large pooled set); slow; GPU;
  hyperparameter-sensitive; poor interpretability — a black box you must
  explain to a bakery manager; heavy serving footprint; **consistently loses to
  GBDT on tabular retail forecasting benchmarks** (see the M5 competition,
  where gradient boosting dominated).
- **Verdict: reject for this problem.** Reconsider only with several years of
  data across hundreds of stores.

### Temporal Fusion Transformer
- **Pros:** genuinely state-of-the-art for multi-horizon multi-series
  forecasting; **native quantile output** (aligns beautifully with the
  newsvendor framing); principled handling of static / known-future / observed
  covariates; built-in attention-based interpretability; strong cold start via
  learned entity embeddings.
- **Cons:** very high complexity; GPU for training; long training times
  (hours vs. minutes) which makes nightly retrain and 30-fold walk-forward CV
  painful; large artifacts; a small team will struggle to maintain
  `pytorch-forecasting` in production; needs a lot of data to beat a
  well-featured LightGBM — and on datasets your size it will typically *lose*
  to it.
- **Verdict: the right long-term aspiration, the wrong choice now.** Revisit at
  ~100+ stores with 3+ years of history. Design the feature store so this
  migration is possible; do not attempt it in year one.

## 3.3 Comparison table

| Model | Accuracy (your data) | Data need | Compute | Interpret | Scale | Realtime | Online | FastAPI | Prod fit |
|---|---|---|---|---|---|---|---|---|---|
| **LightGBM** | **High** | Low | **Low** | Good | **Excellent** | Yes | Warm-start | **Trivial** | **Best** |
| CatBoost | High | Low | Medium | Good | Excellent | Yes | Partial | Easy | Strong alt |
| XGBoost | High | Low | Med-Low | Good | Excellent | Yes | Partial | Trivial | Strong alt |
| Random Forest | Medium | Low | Medium | Good | Good | Yes | No | Easy | Baseline |
| Prophet | Med-Low | Med | High(agg) | **Excellent** | Poor | Slow | No | Awkward | Baseline |
| River | Low-Med | Any | Very low | Medium | Good | **Excellent** | **Yes** | Easy | Drift only |
| ARIMA | Low | Medium | Medium | Good | Poor | No | No | Awkward | Reject |
| SARIMA | Low-Med | High | High | Good | Poor | No | No | Awkward | Reject |
| LSTM | Medium | **Very high** | High(GPU) | **Poor** | Medium | Yes | Partial | Hard | Reject |
| TFT | High(at scale) | **Very high** | **Very high** | Medium | Medium | Yes | No | Hard | Future |

## 3.4 Direct answers

1. **Best for small datasets:** LightGBM as a *global pooled* model — pooling
   is what rescues you from small per-series data. If you truly have one short
   series, Prophet or a seasonal-naive baseline.
2. **Best seasonality handling:** Prophet handles *smooth* seasonality most
   elegantly, but **LightGBM with explicit calendar features handles *your*
   seasonality best**, because your seasonality is lunar + Gregorian + school +
   Coptic simultaneously, which only an explicit-feature approach can express.
3. **Supports online learning:** River (true online); LightGBM/XGBoost/CatBoost
   offer warm-start approximations. **But you should not use online learning.**
4. **Easiest to maintain:** LightGBM — one artifact, one codepath, fast
   retrain, SHAP explanations, no GPU.
5. **Scales best:** LightGBM / CatBoost as global models. (TFT scales
   *statistically* at very large N but not operationally for a small team.)
6. **Best for production:** LightGBM, quantile objective, global.
7. **Best for MVP:** LightGBM — *and before it*, ship a seasonal-naive
   baseline (median of the last 4 same weekdays × calendar multiplier).
   It is one afternoon of work, it will beat a rushed ML model, and it gives
   you the number every future model must beat. **Ship the baseline in week 2.**

---

# Phase 4 — Evaluation Strategy

## 4.1 Splitting — never random

**Random splits are invalid here** and will produce backtests that are inflated
by a large margin. Rows from the same week appear in train and test; the model
effectively interpolates rather than forecasts.

**Recommended: rolling-origin walk-forward validation.**

```
Fold 1: train [2024-01 .. 2024-09]  gap  test [2024-10]
Fold 2: train [2024-01 .. 2024-10]  gap  test [2024-11]
Fold 3: train [2024-01 .. 2024-11]  gap  test [2024-12]
...
Final holdout: the most recent 4–8 weeks, touched exactly once, at the end.
```

- **Expanding window** (as drawn) if data is scarce — uses everything.
- **Sliding window** (fixed-length train) if you suspect regime change, which
  Egyptian inflation makes plausible. Test both; let the data decide.
- **The `gap`** between train end and test start must equal the forecast
  horizon plus label delay. Without it, `lag_1` in the test set uses information
  that would not exist at prediction time. Use `sklearn`'s `TimeSeriesSplit(gap=)`
  or hand-roll it.
- **Fold count:** ≥ 8 folds. A single split on food data is noise —
  one fold landing on Eid will swing your reported metric wildly.
- **Report the metric distribution across folds, not just the mean.** A model
  with 18% mean MAPE and 5% std is far better than one with 16% mean and 20%
  std, because the second one has a catastrophic month in it.

## 4.2 Metrics — and why the requested list is insufficient

| Metric | Verdict |
|---|---|
| **MAE** | **Use.** Interpretable ("off by 12 units on average"), robust, scale-dependent (fine per-product) |
| RMSE | Use with care. Penalises large errors quadratically — good if a big miss is disproportionately bad (often true on Eid), but it is dominated by your highest-volume SKUs, hiding long-tail failure |
| **MAPE** | **Use only for high-volume items.** Mathematically undefined at zero demand and explodes near it — with a zero-inflated bakery catalogue this makes MAPE actively misleading. Also asymmetric: it penalises over-forecasting more than under-forecasting, which is backwards for a bread product |
| SMAPE | Better bounded than MAPE but still unstable near zero and has its own asymmetry quirks. Marginal improvement |
| **R²** | **Do not use.** It answers "does the model beat predicting the global mean," which is a trivially low bar for seasonal data and communicates nothing actionable. It can look excellent while the model is commercially useless |

### Metrics the brief omits and you actually need

**1. Pinball loss (quantile loss) — the primary technical metric.**
If you forecast quantiles (and you should), pinball loss is the correct proper
scoring rule. It is the only metric on this page that directly measures
whether your τ=0.7 forecast really is the 70th percentile.

**2. MASE (Mean Absolute Scaled Error) — the primary comparison metric.**
`MASE = MAE(model) / MAE(seasonal_naive)`. Scale-free, defined at zero,
aggregates safely across products of wildly different volume, and has an
unambiguous interpretation: **MASE < 1 means you beat the naive baseline;
MASE ≥ 1 means the ML project has produced nothing of value.** This is the
number to put in front of leadership.

**3. Bias (Mean Error, signed).** `mean(pred − actual)`. Tells you *direction*.
Every accuracy metric above is symmetric in sign and will hide a model that
systematically under-forecasts by 8% — which, per Challenge 3, is the failure
mode that quietly destroys the product. **Monitor bias per store, per category,
weekly.**

**4. Coverage.** Do the 80% intervals actually contain the outcome 80% of the
time? Miscalibrated intervals make the whole newsvendor policy layer wrong.

**5. The business metrics — these are what actually matter.**
```
EGP_waste_cost   = Σ max(0, produced − demand) × unit_cost
EGP_lost_margin  = Σ max(0, demand − produced) × unit_margin
Total_cost       = EGP_waste_cost + EGP_lost_margin      ← minimise this
Waste_rate       = waste_units / produced_units
Service_level    = 1 − (stockout_days / open_days)
```
Report `Total_cost` for: (a) the model, (b) the seasonal-naive baseline,
(c) **what managers actually did historically**. That third comparison is the
only one the business genuinely cares about, and it is your ROI story.

## 4.3 Which metric to prioritise

- **Model selection & tuning:** pinball loss at your target τ.
- **Cross-model comparison:** MASE.
- **Health monitoring:** bias + coverage.
- **Executive reporting:** EGP total cost avoided vs. manager baseline.
- **Per-product diagnostics:** MAE, and MAPE only where mean demand > ~20/day.

## 4.4 Why food demand is asymmetrically sensitive to overprediction

Restating Challenge 2 in evaluation terms:

**Overproduce by 1 unit:** you lose the full COGS (minus salvage) — realised
immediately, visible, measurable, and in fresh bakery it is a **100% loss** the
same day. It also has downstream costs: disposal, storage, staff time, and, for
a business whose stated mission is reducing waste, reputational cost.

**Underproduce by 1 unit:** you lose the gross margin on a sale — *if* the
customer doesn't substitute. Often they do, recovering much of it. But there is
a tail risk: a customer who finds an empty shelf three times stops coming, and
lifetime-value loss dwarfs a single margin.

So the asymmetry is **real, product-specific, and runs in both directions**
depending on shelf life, margin, and substitutability. This is exactly why a
single symmetric metric (or a single point forecast) cannot express the
objective, and why the τ-quantile framing is the right architecture: **it puts
the asymmetry in an explicit, per-product, auditable business parameter instead
of burying it in a loss function nobody can inspect.**

Practical consequence: expose `Cu` and `Co` (or a simple "how bad is running
out vs. throwing away?" slider) in the product settings UI. Managers understand
this trade-off intuitively. Let them own it.

---

# Phase 5 — AI Architecture

## 5.1 Challenge: seven microservices is the wrong architecture for this system

The brief asks for seven services. I'd push back, and here's the reasoning
rather than just the objection.

**What microservices buy you:** independent scaling, independent deployment,
team autonomy, fault isolation, polyglot freedom.

**What you'd actually have:** all seven services in Python, deployed by one
team, sharing one feature schema, one product catalogue, and one model registry.
Six of the seven have effectively identical load profiles (near-zero, spiking
nightly). "Feature Engineering Service" and "Forecast Service" would exchange
feature vectors over HTTP on every single request — adding latency,
serialisation cost, a network failure mode, and a **distributed training/serving
skew risk**, in exchange for nothing.

**What it costs you:** 7 repos or a monorepo with 7 build pipelines, 7
Dockerfiles, 7 sets of health checks, distributed tracing to debug a single
forecast, schema-version coordination across service boundaries, and — most
expensively — a team spending its first two months on infrastructure instead of
on the waste-capture UX that actually determines whether the project works.

### Recommended: modular monolith + async worker + shared feature library

```
┌────────────────────────────────────────────────────────┐
│  ai-platform  (one FastAPI app, one deployable)         │
│                                                          │
│   /forecast/*   /waste/*   /surplus/*   /model/*  /health│
│                                                          │
│   modules/  forecasting/  waste/  surplus/  registry/    │
│             ^ strict module boundaries, no cross-imports │
│               except via defined interfaces              │
└──────────────┬─────────────────────────────┬────────────┘
               │                              │
      ┌────────▼────────┐          ┌─────────▼──────────┐
      │  features/      │          │  Celery / ARQ      │
      │  (shared lib —  │◄─────────┤  worker            │
      │   ONE codepath  │          │  · nightly batch   │
      │   train+serve)  │          │  · retraining      │
      └────────┬────────┘          │  · drift checks    │
               │                    └─────────┬──────────┘
      ┌────────▼─────────────────────────────▼──────────┐
      │ Postgres (+TimescaleDB)  │ Redis  │ MLflow │ S3  │
      └───────────────────────────────────────────────────┘
```

Keep module boundaries **strict and enforced** (import-linter, or separate
Python packages in a monorepo). Then, if and when a module genuinely needs
independent scaling, extracting it is a contained refactor. **This is the
"start with a monolith, extract services when you have evidence" pattern**, and
for a team of this size with this load profile it is correct.

**The one legitimate early extraction:** the training worker. It has a genuinely
different resource profile (CPU/RAM-heavy, batch, long-running) and different
failure semantics from the request-serving API. Separate that from day one.

## 5.2 If you go with microservices anyway — the design

Presented as the brief requested, since you may have organisational reasons.

### 1. Forecast Service
- **Responsibilities:** load the active model from the registry; assemble
  features (via the shared library or a feature-store read); produce quantile
  forecasts; apply the newsvendor policy layer; cache; log every prediction for
  later scoring.
- **Endpoints:** `POST /forecast/daily`, `/forecast/weekly`, `/forecast/bulk`,
  `GET /forecast/{id}/explain`
- **Depends on:** Model Registry, Feature Service, product catalogue
- **DB:** Postgres (`predictions` table — every prediction logged with model
  version, features hash, and timestamp; this is non-negotiable for later
  accuracy measurement); Redis (hot forecast cache, TTL to next 02:00 batch)
- **Queue:** consumes `bulk_forecast_requested`; publishes `forecast_ready`
- **Scaling:** stateless, horizontal, HPA on RPS. 2–10 replicas. In practice
  most reads hit the nightly-batch cache, so this is very cheap.

### 2. Feature Engineering Service
- **Responsibilities:** the single source of truth for feature computation;
  point-in-time-correct historical features for training; low-latency
  online features for serving; the Egyptian calendar table.
- **Endpoints:** `POST /features/compute`, `POST /features/historical`,
  `GET /features/calendar/{date_range}`, `GET /features/schema/{version}`
- **DB:** TimescaleDB/Postgres for the offline store; Redis for the online
  store. Feature definitions versioned in git.
- **Scaling:** the historical/backfill path is batch and CPU-heavy — scale it
  separately from the online read path.
- **⚠ The critical constraint:** whether this is a service or a library, the
  training path and the serving path **must execute the same code**. If they
  diverge, you get training/serving skew, which produces a model that backtests
  beautifully and fails in production, and it is genuinely hard to diagnose.
  A shared, versioned library is the safer way to guarantee this; a service
  boundary makes it *easier* to accidentally violate.

### 3. Model Training Service
- **Responsibilities:** scheduled and on-demand retraining; walk-forward
  validation; hyperparameter search; log to MLflow; **automated promotion gate**.
- **Endpoints:** `POST /training/jobs`, `GET /training/jobs/{id}`,
  `DELETE /training/jobs/{id}`, `GET /training/history`
- **DB:** Postgres (job state), S3/MinIO (artifacts), MLflow (experiments)
- **Queue:** consumes `retrain_requested` (cron, drift alert, or manual);
  publishes `model_trained`
- **Scaling:** KEDA-scaled Kubernetes Jobs, scale-to-zero. You retrain nightly;
  paying for idle capacity 23 hours a day is waste of a different kind.
- **Promotion gate (important):** a newly trained model is promoted **only if**
  it beats the incumbent on the last N walk-forward folds by a threshold, has
  acceptable bias, and passes calibration checks. Otherwise it is archived and
  an alert fires. **Never auto-promote on training completion alone.**

### 4. Waste Detection Service
- **Responsibilities:** compare a manager's entered quantity against the model
  distribution; compute a risk score; apply configurable per-store thresholds;
  suggest a safer quantity with a stated reason.
- **Endpoints:** `POST /waste/check`, `GET /waste/thresholds/{store_id}`,
  `PUT /waste/thresholds/{store_id}`, `GET /waste/history`
- **Logic:** where does the manager's `q` fall in the predicted distribution?
  `q > P90` → over-production warning. `q < P30` → stockout warning. Severity
  scales with the EGP at risk, not the percentage — a 50% overshoot on a
  10-unit item is not worth an alert.
- **Scaling:** trivial, stateless.
- **UX note that determines success:** alert fatigue kills this feature. Default
  the thresholds *conservatively* (alert rarely), make them per-store tunable,
  and track the **alert action rate**. If managers dismiss > 70% of alerts, the
  thresholds are wrong and the feature is training people to ignore you.

### 5. Surplus Detection Service
- **Responsibilities:** identify stagnant inventory; estimate P(unsold by
  close); recommend discount timing and depth.
- **Endpoints:** `POST /surplus/check`, `GET /surplus/stagnant/{store_id}`,
  `POST /surplus/discount-recommendation`
- **Modelling:** this is a distinct problem from daily forecasting. Given
  `remaining_stock`, `hours_to_close`, and the typical intraday sales curve for
  this `(product, dow)`, estimate `P(sell remaining | time left)`. A
  **survival-analysis or simple hazard-rate framing** fits this better than
  regression. **This is where Isolation Forest belongs** — flagging inventory
  behaving unlike its own history.
- **DB:** needs near-real-time stock, so it reads from the operational DB or a
  CDC stream, not the nightly warehouse.
- **Scaling:** spiky — load concentrates in the 2–3 hours before close, across
  all stores at once. Size for that peak.

### 6. Notification Service
- **Responsibilities:** deliver alerts (push / SMS / WhatsApp / in-app);
  deduplicate; respect quiet hours; batch digests; handle Arabic templates
  and RTL; track delivery and engagement.
- **Endpoints:** `POST /notifications/send`, `GET /notifications/{user_id}`,
  `PUT /notifications/preferences`
- **Queue:** consumes all alert events. **Must be async with retry + DLQ** —
  a WhatsApp API outage must never block a forecast.
- **Egypt note:** WhatsApp is the dominant business-communication channel.
  Prioritise the WhatsApp Business API over email, which will be ignored.
- **Scaling:** queue-depth-based.

### 7. Model Registry Service
- **Responsibilities:** version and store artifacts; stage transitions
  (staging → production → archived); lineage (which data, which features,
  which code commit produced this model); rollback; A/B traffic assignment.
- **Endpoints:** `POST /models`, `GET /models/{name}/versions`,
  `POST /models/{name}/promote`, `POST /models/{name}/rollback`,
  `GET /models/{name}/production`
- **Recommendation: do not build this.** MLflow Model Registry does all of it,
  is battle-tested, and building your own is pure undifferentiated work. Wrap
  MLflow with a thin internal API if you need custom stage logic.
- **Scaling:** minimal. Reads are cached aggressively.

## 5.3 Data flow — the nightly cycle

```
22:00  Stores close. EOD waste + remaining-stock entry (mobile UI).
23:00  ETL: orders + inventory + waste → curated store. Validation gates.
00:30  Feature computation for tomorrow (calendar known; weather forecast pulled).
01:00  Drift checks (data drift, prediction drift, per-store bias).
02:00  Batch forecast: all (store × product) for D+1..D+7, all quantiles.
       → cached in Redis + persisted to predictions table.
03:00  Weekly (Sundays) / triggered: retrain → walk-forward validate →
       promotion gate → promote or alert.
05:00  Morning production plan pushed to managers (WhatsApp + in-app).
06:00+ Managers review; overrides logged as training signal.
Hourly Surplus checks; intensify from close−3h.
Daily  Score yesterday's predictions against actuals → accuracy dashboard.
```

**Note that almost everything is batch.** The real-time API mostly serves
pre-computed results. This is a significant simplification and you should lean
into it — real-time inference is a requirement you can largely avoid.

## 5.4 Infrastructure recommendations

- **Postgres + TimescaleDB** — one database for operational data, features, and
  predictions. Timescale's hypertables and continuous aggregates handle the
  time-series load well without introducing a second datastore. Do not reach
  for a specialised TSDB at this scale.
- **Redis** — online feature cache, forecast cache, rate limiting.
- **Celery or ARQ** (ARQ if you're async-native; Celery for ecosystem maturity)
  with **Redis** as broker. **Do not deploy Kafka.** At your volume it is a
  large operational burden with no payoff. Revisit past ~100 stores or if you
  need event replay.
- **MLflow** — experiments + registry. Self-hosted is fine.
- **S3 / MinIO** — artifacts and Parquet.
- **Feature store:** **do not adopt Feast in v1.** A well-designed shared
  feature library plus Postgres/Redis tables gives you 90% of the value.
  Introduce a real feature store when you have multiple models and multiple
  consuming teams — i.e. when the coordination problem it solves actually
  exists.

---

# Phase 6 — API Design

## 6.1 Conventions

- Base: `/api/v1` — version in the path; breaking changes get `/v2`.
- Auth: JWT bearer; scopes `forecast:read`, `model:write`, `admin`.
  Multi-tenant: **`store_id` access is enforced from token claims, never from
  the request body** — otherwise any tenant can read any other's forecasts.
- All timestamps ISO-8601 with offset; business dates as `YYYY-MM-DD` in
  `Africa/Cairo`.
- Idempotency: `Idempotency-Key` header on all POSTs that mutate or enqueue.
- Correlation: `X-Request-ID` echoed and logged throughout.
- Rate limits per tenant; `429` with `Retry-After`.
- Pydantic v2 models for every request and response. No untyped dicts.

## 6.2 Endpoints

### `POST /api/v1/forecast/daily`

Request:
```json
{
  "store_id": "str-cairo-maadi-01",
  "target_date": "2026-07-21",
  "product_ids": ["prd-baladi-bread", "prd-croissant-plain"],
  "quantiles": [0.1, 0.5, 0.9],
  "include_explanation": true,
  "model_version": null
}
```
Validation: `store_id` exists and is in token scope; `target_date` within
`[today, today+14]` (reject beyond — accuracy is not defensible); `product_ids`
1–500, all active for that store, empty ⇒ all active; `quantiles` ⊂ (0,1),
max 5; `model_version` null ⇒ current production.

Response `200`:
```json
{
  "request_id": "req_01J...",
  "store_id": "str-cairo-maadi-01",
  "target_date": "2026-07-21",
  "model_version": "lgbm-quantile-v14",
  "generated_at": "2026-07-20T02:14:33+02:00",
  "forecasts": [
    {
      "product_id": "prd-baladi-bread",
      "predicted_quantity": 340,
      "quantiles": {"0.1": 265, "0.5": 340, "0.9": 428},
      "recommended_production": 402,
      "recommended_basis": "newsvendor tau=0.67 (Cu=2.40, Co=1.20 EGP)",
      "confidence_score": 0.82,
      "confidence_label": "high",
      "unit": "pieces",
      "explanation": {
        "baseline": 285,
        "drivers": [
          {"feature": "day_of_week=Tuesday", "impact_pct": -4.2},
          {"feature": "days_since_payday=2",  "impact_pct": +12.8},
          {"feature": "rolling_mean_7",       "impact_pct": +8.1}
        ],
        "manual_adjustments": []
      },
      "data_quality": {
        "history_days": 214,
        "censored_days_28": 3,
        "is_cold_start": false
      }
    }
  ],
  "warnings": []
}
```

**Design notes worth defending:**
- `confidence_score` is deliberately **derived from interval width relative to
  the median**, not invented. Define it once:
  `confidence = 1 − clip((P90−P10) / (2·P50), 0, 1)`. Also surface
  `confidence_label` — managers will not act on 0.82, they will act on "high".
- `recommended_production` ≠ `predicted_quantity`. The former is the business
  decision (quantile + policy), the latter is the statistical estimate.
  **Keeping these separate and both visible is the most important API design
  choice here.**
- `data_quality` on every forecast so the client can visually degrade
  cold-start or heavily-censored predictions rather than presenting them with
  false authority.

### `POST /api/v1/forecast/weekly`
Same shape, `start_date` + `horizon_days` (1–14), returns a per-day array plus
`week_total` and a `production_schedule` grouped by day. Note that weekly
forecasts should be generated by a **direct multi-horizon model** (a separate
model per horizon, or horizon-as-a-feature), **not** by recursively feeding
day-1 predictions back in — recursive forecasting compounds error badly.

### `POST /api/v1/forecast/bulk`
Async by design. Accepts up to 50 stores × all products. Returns `202 Accepted`
with `{job_id, status: "queued", poll_url, estimated_completion}`. Results via
`GET /forecast/bulk/{job_id}` (paginated) or a webhook. **Never make this
synchronous** — it will time out and retries will stampede the worker.

### `POST /api/v1/forecast/retrain`
`202` with `job_id`. Body: `{scope: {store_ids?, product_categories?},
trigger_reason, force: bool, hyperparameter_search: bool}`.
Requires `model:write`. Rejects with `409` if a job is already running for the
same scope. Result includes validation metrics and the **promotion decision with
its reason** — visible, not implicit.

### `POST /api/v1/waste/check`
```json
{"store_id":"...", "target_date":"2026-07-21",
 "entries":[{"product_id":"prd-croissant-plain","planned_quantity":500}]}
```
Response:
```json
{"results":[{
  "product_id":"prd-croissant-plain",
  "planned_quantity": 500,
  "forecast_median": 180,
  "forecast_p90": 240,
  "percentile_of_plan": 99.4,
  "risk_level": "high",
  "expected_waste_units": 315,
  "expected_waste_value_egp": 4725.0,
  "suggested_quantity": 214,
  "suggested_basis": "newsvendor tau=0.61",
  "potential_saving_egp": 4290.0,
  "message_ar": "الكمية المخططة أعلى بكثير من الطلب المتوقع",
  "message_en": "Planned quantity is far above expected demand"
}]}
```
Lead with **EGP**, not percentages. "You may waste 4,725 EGP" changes behaviour;
"178% above forecast" does not.

### `POST /api/v1/surplus/check`
Input: current stock, current time, hours to close. Output per product:
`probability_unsold`, `expected_leftover_units`, `expected_loss_egp`,
`recommended_action` (`none` | `discount` | `bundle` | `staff_meal` | `donate`),
`recommended_discount_pct`, `optimal_action_time`, `urgency`.

The discount recommendation should maximise
`expected_revenue(discount) − expected_waste_cost(discount)`,
not simply "discount when at risk." A 50% discount that would have sold at full
price anyway destroys margin — cannibalisation is the real risk in this feature.

### `GET /api/v1/model/status`
Active version(s), trained-at, training data range, feature-schema version,
health (`healthy`/`degraded`/`stale`), days since retrain, drift status,
A/B split if active, rollback availability.

### `GET /api/v1/model/metrics`
Query: `?model_version=&store_id=&from=&to=&granularity=`.
Returns MAE, RMSE, MASE, pinball loss per τ, **bias**, coverage, plus the
business block (`waste_cost_egp`, `lost_margin_egp`, `total_cost_egp`) and
the **baseline comparison** (`vs_seasonal_naive`, `vs_manager_baseline`).

### `GET /health`
Two endpoints, distinct semantics:
- `GET /health/live` — process is up. No dependency checks. For k8s liveness.
- `GET /health/ready` — DB, Redis, registry reachable and a model is loaded.
  For k8s readiness and load-balancer membership.

Conflating these is a classic outage cause: a slow DB fails the liveness probe,
k8s restarts every pod, and you turn a degradation into an outage.

## 6.3 Error handling

RFC 7807 `application/problem+json`:
```json
{
  "type": "https://api.example.com/errors/insufficient-history",
  "title": "Insufficient history for forecast",
  "status": 422,
  "detail": "Product prd-new-item-99 has 4 days of history; 14 required for the standard model.",
  "instance": "/api/v1/forecast/daily",
  "request_id": "req_01J...",
  "context": {"product_id": "prd-new-item-99", "history_days": 4, "required": 14},
  "fallback_applied": "category_average",
  "fallback_result": {"predicted_quantity": 45, "confidence_score": 0.21}
}
```

| Code | When |
|---|---|
| 400 | Malformed request |
| 401 / 403 | Unauthenticated / store not in token scope |
| 404 | Unknown store or product |
| 409 | Conflicting retrain job in flight |
| 422 | Valid syntax, unsatisfiable semantics (horizon too long, no history) |
| 429 | Rate limited |
| 500 | Unexpected — never leak internals |
| 503 | No production model loaded / dependency down |

**Degradation ladder — the most important error-handling decision.** A forecast
API should almost never return a hard error, because a manager at 06:00 needs
*a number*. Cascade:
```
1. Current production model
2. Previous model version (if current fails to load)
3. Cached forecast from the last successful batch (flagged stale)
4. Statistical fallback: median of last 4 same weekdays × calendar multiplier
5. Category average scaled by store size
6. Only now: 503
```
Every response carries `fallback_applied` and a lowered `confidence_score`, so
the UI can be honest about degradation rather than silently serving a worse
number at full confidence.

---

# Phase 7 — MLOps & Deployment

## 7.1 Docker

Multi-stage, non-root, slim. Three images from one repo:
```
Dockerfile.api      → FastAPI + uvicorn/gunicorn. No training deps. ~300MB
Dockerfile.worker   → Celery/ARQ + LightGBM + pandas. ~800MB
Dockerfile.trainer  → + MLflow, Optuna, full validation stack. ~1.2GB
```
Shared base layer for common deps. Pin everything with `uv` or
`pip-compile` — a silent LightGBM minor bump can shift predictions.
`.dockerignore` the data directory. Health checks in the image.
Trivy scan in CI.

## 7.2 Kubernetes

```
Deployment  ai-api        2–10 replicas, HPA on RPS+CPU, PDB minAvailable=1
Deployment  ai-worker     2–8 replicas, KEDA on Redis queue depth
CronJob     nightly-batch    02:00 Africa/Cairo
CronJob     retrain          Sun 03:00
CronJob     drift-check      01:00 daily
CronJob     score-predictions 04:00 daily (yesterday's accuracy)
Job         training-run  on-demand, KEDA scale-to-zero, 8Gi/4CPU, TTL cleanup
```
Rolling updates for the API; `Recreate` for stateful workers. Resource requests
set from observed p95, limits at ~2× requests. Secrets via External Secrets, not
env-var literals. Namespace per environment.

**Honest scoping note:** if you are under ~10 stores, **Kubernetes is
over-engineering**. Docker Compose on two decent VMs, or ECS/Cloud Run, will
serve you fine and cost a fraction of the operational attention. Adopt k8s when
you have a platform engineer, not before. The architecture above survives that
choice — the containers are the same.

## 7.3 CI/CD

```
PR:        lint (ruff) → format → mypy → unit tests → feature-pipeline tests
           → data-contract tests → build → Trivy → deploy to ephemeral env
main:      integration tests → model smoke test (train tiny, assert it beats naive)
           → deploy staging → shadow traffic 24h → manual gate → canary 10%
           → monitor 1h → full rollout
nightly:   full walk-forward validation on latest data → post metrics to Slack
```

**Model CD is separate from code CD, and this distinction matters.** A new
model version does not require a new deployment — the API loads the production
model from the registry. Model promotion is a registry stage transition, gated
on metrics, and rollback is a stage transition back. This decoupling is what
lets you retrain nightly without deploying nightly.

**Test the feature pipeline harder than you test the API.** A regression in a
lag calculation is silent, expensive, and will not be caught by any HTTP test.
Golden-dataset tests: fixed input → asserted feature values.

## 7.4 Retraining

- **Scheduled:** weekly full retrain (Sunday 03:00). Nightly is unnecessary —
  demand patterns don't shift that fast, and weekly halves your compute and
  your churn.
- **Triggered:** data drift beyond threshold; sustained bias > 10% for 7 days;
  MASE degradation > 15% vs. the promotion-time value; a new store or a bulk
  catalogue change.
- **Always** through the promotion gate. Never auto-promote on completion.
- **Retrain window:** test expanding vs. sliding (24-month) empirically.

## 7.5 Versioning & tracking

Version four things independently, and record all four on every prediction:
**code** (git SHA), **data** (snapshot ID / date range hash), **features**
(schema version), **model** (registry version). Without all four you cannot
reproduce a prediction, and "why did it say 500 on Eid?" becomes unanswerable.

MLflow for experiments: log params, metrics per fold, feature importance,
the walk-forward plot, and the data hash. Optuna for HPO, with a strict trial
budget — the returns are small relative to better features.

## 7.6 Monitoring

**System:** latency p50/p95/p99, error rate, throughput, queue depth, worker
saturation, batch-job success and duration. Prometheus + Grafana.

**Data:** freshness, null rates, row counts, schema violations, quarantine
rate, **waste-capture compliance per store** (this one is a business health
metric disguised as a data metric — watch it weekly).

**Model:**
- *Data drift* — PSI / KS per feature vs. the training reference. PSI > 0.2 =
  investigate, > 0.25 = alert. Expect and **suppress** the seasonal false
  positives: Ramadan will trip every drift detector you own. Compare against a
  seasonally-matched reference window, not a fixed one.
- *Prediction drift* — distribution of outputs vs. recent history. Catches
  breakage before labels arrive, which is its whole value given label delay.
- *Concept drift* — the real one: accuracy degradation on scored predictions.
  Lags by one day; alert on a 7-day rolling MASE increase.
- *Bias monitoring* — signed error per store and per category, weekly. **This
  is your early warning for the under-forecast doom spiral.**
- *Calibration* — realised coverage of the 80% interval.

Evidently AI or NannyML for the drift reports; both integrate cleanly and save
you writing this yourself.

**Business dashboard (the one leadership sees):** EGP waste avoided, waste rate
trend, service level, forecast-vs-manager comparison, alert action rate,
recommendation adoption rate.

## 7.7 Logging

Structured JSON (`structlog`), correlation IDs propagated through the queue,
**every prediction persisted** with inputs hash + model version + timestamp.
Log manager overrides with the delta — your richest feedback signal.
No PII in logs. Retain predictions ≥ 2 years (you need them for year-over-year
features and for audit).

## 7.8 The four operating modes

- **Batch** (primary): nightly, covers ~95% of usage.
- **Real-time**: on-demand recalculation after a manager changes an input.
  Cheap with a global GBDT — a few ms.
- **Online learning**: **recommend against it.** Nightly/weekly batch retrain
  gives you the adaptivity with full reproducibility, and reproducibility is
  worth more than the marginal freshness. Revisit only if you observe genuine
  intraday regime shifts, which you almost certainly won't.
- **A/B testing**: assign by `store_id` hash (**not by request**, or the same
  store gets inconsistent numbers day to day and managers lose trust). Minimum
  2 weeks to cover weekly seasonality — ideally 4. Primary metric: EGP total
  cost, not MAE. Guardrail: service level must not degrade.

---

# Phase 8 — Final Recommendation

## 8.1 Best architecture

**Modular monolith (FastAPI) + async worker + shared feature library**, on
Postgres/TimescaleDB + Redis + MLflow + S3, deployed as containers. Extract
services only on evidence of a real scaling or team-boundary need. Batch-first;
real-time as a thin layer over cached results.

## 8.2 Best MVP model

**Stage 0 (week 2): seasonal-naive baseline.** Median of the last 4 same
weekdays × Egyptian calendar multiplier. Ship it. It is genuinely useful, it
takes a day, and it is the bar every later model must clear.

**Stage 1 (week 6): LightGBM, global, quantile objective**, trained across all
`(store × product)` series with the ~25 features from §2.7, three quantiles
(0.1 / 0.5 / 0.9), plus the newsvendor policy layer.

## 8.3 Best long-term model

The same LightGBM, matured: more quantiles, hierarchical reconciliation
(store → category → product totals should be coherent), a dedicated
intermittent-demand path for the long tail, and censoring-aware training.
TFT only past ~100 stores and 3 years of data, and only with dedicated ML
engineering capacity.

## 8.4 Best online-learning model

**None — and this is a recommendation to not build a requested feature.**
Use nightly/weekly batch retraining. If you want online *adaptation*, add a
lightweight residual-correction layer (an EWMA of recent per-product bias
applied as a multiplier), which gives you most of the responsiveness with none
of the reproducibility loss. Use River for **drift detection** only.

## 8.5 Best anomaly detection strategy

**Layered, and forecast-residual-based rather than raw-feature-based:**

1. **Rule-based** (deterministic, explainable, catches most real problems):
   negative quantities, identity violations, impossible values.
2. **Forecast-residual** (primary): flag when
   `|actual − predicted| > k × predicted_std`. This is superior to a generic
   detector because it is *conditional on context* — an Eid spike produces a
   small residual (the model expected it) while a genuine anomaly produces a
   large one. **A generic Isolation Forest on raw features would flag Eid and
   miss the anomaly. This is the key insight of this section.**
3. **Isolation Forest** for the multivariate surplus/stagnant-inventory case,
   where there is no forecast to residual against.
4. **Survival/hazard model** for "will this sell before close."
5. **Business-threshold layer** on top: only alert when the EGP at risk exceeds
   a per-store floor. Statistical significance is not the same as business
   significance, and confusing them is how you produce an alert system nobody
   reads.

## 8.6 Best FastAPI architecture

Single app, `/api/v1`, routers per domain, Pydantic v2 everywhere, async
endpoints with sync model inference dispatched to a thread pool (LightGBM
inference is CPU-bound and will block the event loop otherwise — a very common
and very confusing performance bug). Model loaded once at startup into app
state, hot-swappable via a registry-watch endpoint. Dependency injection for
DB/Redis/registry. Split liveness and readiness probes. RFC-7807 errors with
the degradation ladder. Auto-generated OpenAPI as the contract with the
e-commerce platform team.

## 8.7 Implementation complexity

| Component | Complexity | Est. effort |
|---|---|---|
| Data capture instrumentation (waste, production, sellout) | **Medium — but organisationally hard** | 3–4 wks |
| Cleaning + validation pipeline | Medium | 2 wks |
| Egyptian calendar table | Low technically, high domain-research | 1 wk |
| Feature pipeline | Medium-High (correctness-critical) | 3 wks |
| Baseline model | Low | 3 days |
| LightGBM quantile + newsvendor layer | Medium | 2 wks |
| Evaluation harness (walk-forward) | Medium | 1 wk |
| FastAPI service | Medium | 2 wks |
| Waste + surplus features | Medium | 2 wks |
| MLOps (CI/CD, registry, monitoring) | High | 3 wks |
| Manager UI integration | Medium | 3 wks |
| **Total to production MVP** | | **~14–16 weeks, 2–3 engineers** |

**The hardest part is not the model. It is getting managers to reliably record
waste every single night.** Budget more attention there than the estimate above
implies.

## 8.8 Estimated accuracy — realistic, not aspirational

Assuming 12+ months of clean, Egyptian, per-product daily data:

| Segment | MAPE | MASE | Notes |
|---|---|---|---|
| High-volume staples (bread, top pastries) | **12–20%** | 0.65–0.80 | The reliable win |
| Mid-volume (20–100/day) | 25–40% | 0.75–0.90 | Usable |
| Long tail (< 10/day, intermittent) | 50–90% | 0.90–1.05 | **May not beat naive. Be honest about this** |
| Ramadan / Eid, year 1 | 40–80% | — | Insufficient history; heuristic-driven |
| Ramadan / Eid, year 3 | 20–35% | — | Once 2–3 cycles are observed |
| New products (cold start) | 60–100% | — | Educated prior, not a forecast |

**Expected business outcome, which is the number that matters:**
**25–40% reduction in food waste within 6 months of full adoption**, driven
less by forecast precision than by (a) simply *measuring* waste for the first
time, (b) removing systematic over-production habits, and (c) timely surplus
discounting. Be candid with stakeholders: much of the early gain comes from
measurement and process, not from the model. That is a feature, not an
admission — and it is why the instrumentation milestone comes first.

## 8.9 Roadmap & milestones

**Phase A — Foundation (weeks 1–4)**
- M1 (wk 1): Dataset profiling complete; hypotheses in Phase 0 confirmed or
  refuted in writing. Egyptian calendar table v1 built and reviewed.
- M2 (wk 2): Waste/production/sellout capture shipped in the platform. Pilot
  with 2–3 friendly stores. **Gate: is compliance > 80% after two weeks?**
  If not, stop and fix the UX before writing any model code.
- M3 (wk 4): Cleaning + validation pipeline live; data quality dashboard.

**Phase B — Baseline & first value (weeks 5–8)**
- M4 (wk 5): Seasonal-naive baseline in production, delivering daily
  recommendations. **First real user value.**
- M5 (wk 6): Feature pipeline + walk-forward evaluation harness.
- M6 (wk 8): LightGBM quantile model beats baseline on MASE across ≥ 8 folds.
  **Gate: if it doesn't beat the baseline, the problem is data, not model —
  do not proceed to tuning.**

**Phase C — Productionisation (weeks 9–14)**
- M7 (wk 10): FastAPI service live with the full degradation ladder;
  integrated with the e-commerce platform.
- M8 (wk 12): Waste-check and surplus-detection features live; alerts flowing
  via WhatsApp.
- M9 (wk 14): MLOps complete — automated retraining, drift monitoring,
  registry, promotion gate, rollback tested (actually tested, not assumed).

**Phase D — Scale & refine (months 4–9)**
- M10: A/B test model vs. manager baseline. Publish the EGP result.
- M11: Roll out across all stores; per-store threshold tuning.
- M12: Cold-start improvements; hierarchical reconciliation.
- M13: Weather integration (only if M10 shows the model is the binding
  constraint).
- M14: First full Ramadan/Eid cycle with the system live — **treat as a
  research milestone**, capture everything, expect to be wrong, and build the
  post-mortem into the plan.

**Phase E — Advanced (year 2+)**
Cross-product substitution modelling; price/discount optimisation; ingredient
BOM and procurement forecasting; TFT evaluation; multi-tenant SaaS packaging.

---

# Open questions I need answered

These genuinely change the design; I've stated my working assumption for each
so nothing is blocked.

1. **Do you already capture production quantity and end-of-day waste?**
   *Assumed: no.* If yes, the roadmap shortens by ~4 weeks and Phase A changes
   substantially. This is the single most schedule-relevant question.
2. **How much order history exists in the platform, and across how many stores
   and SKUs?** *Assumed: < 12 months, few stores.* If you have 2+ years, the
   Ramadan/Eid features become learnable rather than heuristic, which changes
   Phase 2's recommendation materially.
3. **Bakeries or restaurants first?** *Assumed: bakeries.* They are the better
   starting point — shorter shelf life, higher waste, more repeatable demand,
   clearer ROI. Restaurants add BOM complexity and per-dish decomposition.
4. **Single business or multi-tenant SaaS?** *Assumed: multi-tenant eventually.*
   This drives tenant isolation in the data model, and retrofitting it is
   expensive — worth deciding early even if you build single-tenant first.
5. **Team size and MLOps maturity?** *Assumed: 2–3 engineers, no dedicated
   platform engineer.* This is why I recommend against Kubernetes, Kafka,
   Feast, and microservices for v1. If you have a platform team, several of
   those recommendations flip.
6. **Is there an existing manager-facing app to put recommendations in?**
   *Assumed: yes, the e-commerce admin.* If not, add ~4 weeks.

---

# Appendix — Recommended stack

```
Python 3.12
FastAPI + Pydantic v2 + uvicorn/gunicorn
LightGBM (primary) · scikit-learn · pandas / polars
Optuna (bounded HPO) · SHAP (explanations)
MLflow (experiments + registry)
Postgres 16 + TimescaleDB · Redis 7
Celery or ARQ
Evidently AI (drift) · Prometheus + Grafana · structlog
pandera or Great Expectations (data contracts)
hijri-converter (+ a manually confirmed Egyptian calendar table)
pytest · ruff · mypy · import-linter (enforce module boundaries)
Docker · (Kubernetes only when a platform engineer exists)
```

Deliberately excluded from v1, with reasons: Kafka (no volume justification),
Feast (no multi-consumer coordination problem yet), TFT/LSTM (insufficient data,
insufficient team capacity), River as a forecaster (reproducibility cost
exceeds the benefit), Airflow (k8s CronJobs plus Celery beat suffice at this
scale).
