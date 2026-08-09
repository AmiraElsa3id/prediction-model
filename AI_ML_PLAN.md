# AI/ML POC Plan — Bakery Demand Forecasting & Surplus Marketing (Egypt)

## Context

We are building a **proof of concept** for an AI layer on top of an existing e-commerce/POS
system for Egyptian bakeries, targeting food waste and guess-based production. The audience
is **stakeholders/investors**, so the POC must run end-to-end and tell a convincing product
story. It is explicitly *not* production software.

The governing constraint: **there is no real data.** The project is pre-launch, and the three
Kaggle datasets are unusable for this purpose — one has no date column, one covers only 100
days, and none contain any Egyptian calendar signal (Ramadan, Eid, Coptic holidays, school
terms, Thu/Fri weekend). So we **generate a realistic synthetic Egyptian bakery dataset** and
build against it.

### One integrity rule, non-negotiable

Every chart, metric, and figure in the demo must be **labelled as simulated**. The waste-saved
number is a projection from synthetic data, not a measured result from a real bakery. Investors
being told otherwise is fraud, and it also collapses the moment they ask for the raw data. The
honest framing is stronger anyway: *"here is the system working, and here is what it projects
once connected to a real POS."*

Confirmed with the user: pre-launch, no data; system tracks sales + inventory with end-of-day
waste (so leftovers are derivable); shelf life unknown → configurable table; free-tier LLM for
Arabic; publishing/approval behaviour deferred; all six endpoints in scope.

### Cut from the production plan (deliberately)
Multi-tenant state, Postgres/Redis, model registry, champion/challenger promotion, nightly
retrain jobs, hierarchical cross-tenant priors, deep censoring correction. None of it helps a demo.

---

## 1. What the demo must show

A five-beat narrative — every build task below serves one of these:

1. **"It knows Egypt."** A chart where Ramadan, Eid kahk demand, and the Fri/Sat weekend are
   plainly visible in the forecast. This is the differentiator.
2. **"It predicts tomorrow."** Per-item production quantities with confidence intervals.
3. **"It stops waste before it happens."** Manager types 200 croissants, system warns that
   forecast is 120 and projects the EGP loss.
4. **"It sells the surplus by itself."** 4pm: 40 unsold gateau detected → 30% offer generated
   in Egyptian Arabic → posted.
5. **The money slide.** Simulated year: AI-guided production vs. manual baseline → **% waste
   reduced and EGP saved per month.** This is the number investors remember.

---

## 2. ML framing

Two decisions carry the intellectual weight of the pitch:

**(a) Forecast a quantile, not the mean.** Over-producing loses full COGS; under-producing
loses only margin. The newsvendor optimum `q* = Cu/(Cu+Co)` therefore sits **below** expected
demand for typical bakery economics. A model tuned for MAE will systematically over-produce and
*increase* waste — the exact failure we're selling against. Implemented as one LightGBM
parameter (`objective="quantile", alpha=q`), so it costs almost nothing and is a genuinely
strong talking point: *"we deliberately forecast low, because waste costs more than a lost sale."*

**(b) Sales ≠ demand.** Sell out at 2pm and the data understates true demand; train on it
naively and the model under-produces forever in a self-reinforcing spiral. For the POC, handle
this simply: flag `closing_stock == 0` days and exclude them from the training loss. The full
Tobit-style correction is deferred.

---

## 3. Synthetic data generator — the foundation

`data/generate.py` — 2 years, 1 branch, ~10 real items (عيش بلدي، فينو، كرواسون، جاتوه، كحك،
فطير، بسبوسة …), each with cost/price and shelf life.

Injected with **known** parameters, so we can verify the model recovers them:
- Per-item weekday shape (Fri/Sat uplift)
- Ramadan: daytime suppression + pre-iftar spike, as a day-index curve — **not** a binary flag
- Kahk ramp in the ~10 days before Eid al-Fitr
- Sham El-Nessim feteer spike; school-term morning pastry effect; end-of-month payday bump
- Mild upward trend, noise, occasional stockouts and closed days

Because 2 years exist in synthetic data, batch models (LightGBM/Prophet) train properly and the
demo looks strong — while the cold-start story below covers what happens for a *real* new bakery.

---

## 4. Models — POC scope

| Role | Model | Why |
|---|---|---|
| Baseline (must beat) | Seasonal naive (same weekday last week) | Honest floor; often competitive |
| **Primary** | **LightGBM, quantile objective, lag + calendar features** | Best accuracy, one model all items, <10 ms, native quantile |
| Visual | Prophet | Its **decomposition plot showing the Ramadan hump** is the single best investor visual |
| Cold-start story | River (`HoeffdingAdaptiveTreeRegressor`) | Demonstrates "works from day 1 for a new bakery, no history" |

**Recommendation: LightGBM primary + seasonal-naive baseline as P0; Prophet chart and River
cold-start as P1.** Deep learning (LSTM/DeepAR) is explicitly rejected — with a few dozen SKUs
it would underperform a moving average at far higher cost.

Full comparison table (pros/cons/data/latency) is retained in §9 for the methodology slide.

---

## 5. Egyptian calendar — `egypt_calendar.py`

The long pole, and buildable immediately with no data.
- **Hijri conversion** (`hijri-converter`) for Ramadan + both Eids. These shift ~11 days earlier
  each Gregorian year, so they must be computed, never hardcoded — and must be *injected as
  features*, since no model can infer them from two years of history.
- Coptic calendar → Sham El-Nessim, Coptic Christmas/Easter.
- School terms; Egyptian public holidays (Jan 25, Apr 25, May 1, Jun 30, Jul 23, Oct 6).
- Emits: `is_ramadan`, `ramadan_day_index`, `days_to_eid`, `is_eid`, `is_public_holiday`,
  `is_school_term`, `is_weekend` (**Fri/Sat**), `is_payday_window`.

One module, three consumers: Prophet (custom holidays DataFrame with lower/upper windows),
LightGBM and River (engineered columns).

---

## 6. Evaluation

- **Split:** strictly time-based, rolling-origin (expanding window), 1- and 7-day horizons, with
  a gap to prevent leakage. Lags computed inside the fold. No random splits.
- **Metrics:** WAPE headline (MAPE is unusable — items hit zero, and it punishes over-forecasting
  asymmetrically), MASE vs seasonal naive, pinball loss (matches the quantile objective), MAE/RMSE
  per item.
- **Recovery test:** confirm the model recovers the injected Ramadan/Eid/weekday effects. This is
  the core scientific proof, and it's only possible *because* the data is synthetic.
- **Business simulation:** AI-guided vs. manual production over the simulated year → units wasted,
  stockout rate, **EGP saved**. Feeds the money slide.
- **Cleaning/outliers:** reconcile `production − sales = leftover`; separate "closed" from "zero
  sales"; Hampel/rolling-MAD outlier filter that **never** flattens a spike explained by a calendar
  feature — a pre-Eid surge is signal, not noise.

---

## 7. Service — FastAPI, all six endpoints

Pydantic in/out, Swagger at `/docs`, typed error envelopes. State in local pickle/SQLite — no DB.
Every forecast response returns `confidence`, `source`, prediction interval, and
`contributing_factors` (e.g. `"Ramadan day 12: -35%"`) — the explanation is what makes a manager
trust the number, and it demos far better than a bare integer.

- `POST /forecast/daily`, `POST /forecast/weekly`
- `POST /forecast/seasonality-adjustment` → calendar multipliers + human-readable reason
- `POST /alerts/waste-prevention` → manual qty vs forecast interval → severity + projected EGP loss
- `POST /surplus/detect` → risk score from remaining stock, hours to close, expected sell-through,
  and the configurable shelf-life table
- `GET /health`

**Discount sizing is rule-based tiers by risk score** — price elasticity cannot be learned without
data, and discounting more than needed to clear stock just donates margin.

**Arabic copy:** free-tier LLM (Gemini Flash / Groq / OpenRouter `:free`) behind a swappable
`CopyGenerator` interface, with hand-written Egyptian Arabic templates as automatic fallback if the
API fails or output fails a safety/price check. A live demo must never hang on someone's API.

---

## 8. Build order

**P0 — the demo runs end to end**
1. `egypt_calendar.py` + unit tests (start here; no data needed)
2. Synthetic data generator
3. Feature pipeline + cleaning + stockout flagging
4. Seasonal-naive baseline + backtest harness + metrics
5. LightGBM quantile model
6. FastAPI + Pydantic schemas + all six endpoints (publish mocked)
7. Arabic copy generation + template fallback
8. Business simulation → the EGP-saved number

**P1 — makes the demo land**
9. Prophet decomposition chart (Ramadan hump)
10. River cold-start blend, to tell the "day 1, no history" story
11. Small Streamlit dashboard — Swagger is unconvincing to non-technical investors

**P2 — only if time allows**
12. Real Meta publishing against a throwaway test Page

---

## 9. Verification

- `pytest` on the calendar: assert Ramadan/Eid dates against known 2023–2027 values, Coptic Easter,
  and the ~11-day annual drift.
- Recovery test: assert injected weekday/Ramadan/Eid effects are recovered within tolerance.
- Backtest: LightGBM must beat seasonal naive on WAPE/MASE, or we ship the naive model and say so.
- Stockout test: inject known sell-outs, confirm the model doesn't spiral downward.
- End-to-end: run the service, drive all six endpoints via Swagger, verify JSON shapes.
- Full demo rehearsal: walk the five beats in §1 start to finish.

## 10. Open items (do not block the POC)

Needed only when moving beyond the demo to a real bakery:
1. Confirmation that the inventory schema exposes closing stock + stock-added per item per day.
2. Real shelf life per item/category (POC uses defaults).
3. Real unit economics (cost/price) — POC assumes plausible values to set `q*`.
4. Branch count, SKU count, opening/closing hours.
5. Eventually 12+ months of real POS data, at which point the cold-start blend hands over to
   batch models.
