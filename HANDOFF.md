# HANDOFF — Bakery Demand & Surplus AI

> **Purpose of this file:** a complete, self-contained state dump so any AI or developer
> can read it and continue the work without the prior conversation. Written in English
> for universal readability; the product/UI language is Egyptian Arabic. Last updated
> after building the RestoMind integration bridge + weekly prediction adapter.

---

## 1. What this project is

An **AI/ML layer for Egyptian bakeries & restaurants** that:
1. **Forecasts production demand** per product per day/week, aware of the Egyptian
   calendar (Ramadan, Eid, Coptic feasts, Sham El-Nessim, Fri/Sat weekend, school terms,
   payday) → cuts over-production waste.
2. **Detects end-of-day surplus** and **generates Egyptian-Arabic discount offers**, with
   optional auto-publishing to Facebook/Instagram.

It is a **Python FastAPI microservice**. It is meant to plug into an existing e-commerce/
POS backend (the client's system is **RestoMind**, a NestJS + MongoDB app — see §7).

**Owner context:** pre-launch, no real sales data yet. This is a POC / capstone.
Audience for pitches = an academic committee + a bakery/restaurant client.

### The single most important honesty rule
All accuracy/EGP figures are from **SIMULATED data** (a synthetic generator), not a real
bakery. This is stated in `/health`, the API landing page, the dashboard, and
`problem_analysis.html`. **Never present simulated numbers as measured.** The real proof
requires a pilot (forecast vs actual on a real bakery).

---

## 2. Environment / how to run

- Python **3.13**, venv at `.venv/`. Deps in `requirements.txt` (pandas, numpy, lightgbm,
  scikit-learn, hijridate, convertdate, fastapi, uvicorn, pydantic, httpx, pyarrow,
  streamlit, plotly, pytest, newman via npx for Postman).
- Working dir: `/Users/amera/Desktop/Home/model`
- **Shell note:** `head` is aliased to something broken in this environment — use `sed -n`
  / `grep -m` / Python instead of piping to `head`.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m app.core.generate       # (re)generate synthetic dataset -> data/synthetic_pos.parquet
.venv/bin/python -m pytest tests/ -q         # 79 tests, ~2.5 min (trains real models in API tests)
.venv/bin/python -m scripts.run_backtest     # model comparison table
.venv/bin/python -m scripts.run_simulation   # business simulation (EGP saved)
.venv/bin/uvicorn app.api.main:app           # API + Swagger at /docs (trains on startup, ~20s)
.venv/bin/streamlit run dashboard.py         # investor dashboard
```
Env vars: `COLD_START=true` (start with no history, fully rule-based),
`LLM_API_KEY`/`LLM_BASE_URL`/`LLM_MODEL` (free-tier LLM for Arabic copy + priors),
`META_PAGE_ID`/`META_ACCESS_TOKEN`/`META_PUBLISH_ENABLED` (live publishing),
`REQUIRE_API_KEY`/`API_KEY_HASH` (API key auth, see `docs/01-api-key-hardening.md`; unset in
dev, auth is skipped; `API_KEY_HASH` is a SHA-256 hash, never the raw key),
`RATE_LIMIT_DEFAULT_PER_MIN`/`RATE_LIMIT_MARKETING_PER_MIN`/`RATE_LIMIT_WINDOW_SECONDS`
(cost-guard rate limit overrides, see `docs/03-cors-and-rate-limiting.md`; unset falls
back to 300/20/60 -- not a real per-user limiter, just a ceiling against runaway cost).
No `CORS_ORIGINS` anymore -- this service is never called from a browser directly, so
CORSMiddleware was removed (`docs/03` §2.1) rather than configured.

**Status: 79 tests passing.** Do not mark work done unless tests pass.

---

## 3. Architecture & file map

```
app/
  core/
    egypt_calendar.py   Ramadan/Eid (Hijri via hijridate) + Coptic (convertdate) + school/
                        holiday/payday/weekend features. RAMADAN_OVERRIDES pins declared dates.
    items.py            11-item bakery catalogue: economics, shelf life, newsvendor q*,
                        spoilage_severity, stockout_goodwill_mult. Ground-truth effect sizes.
    generate.py         Synthetic POS+inventory generator (2 yrs) with KNOWN injected effects
                        + a "manager guesses" production policy (creates the waste baseline).
    features.py         Cleaning (reconcile stock, closed-vs-zero, outlier flag that spares
                        calendar spikes), lag/rolling/calendar features, censored-demand mask,
                        clean_baseline (event-free baseline for multiplier fitting).
    evaluation.py       WAPE/MASE/pinball/bias + rolling-origin backtest + per-item scores.
    surplus.py          Near-closing surplus detection + discount tiers + sell-through curve.
    market_priors.py    LLM/market-analysis calendar sensitivities per category+item.
                        load_priors(), category_priors(), generate_priors_llm() (LLM+fallback).
  models/
    seasonality.py      CalendarEffects: per-item calendar multipliers via Ridge in log space
                        (calendar-only design matrix, NO lags). explain() = exact attribution.
    forecaster.py       SeasonalNaive, MovingAverage, LightGBMQuantile, and CalendarDecomposed
                        (THE production model — see §4).
    rule_based.py       RuleBasedForecaster: cold-start using market_priors + owner level.
                        rule_multiplier(priors, feats) -> (mult, factors).
    service.py          ForecastService: HYBRID routing (rule-based <90 days, trained >=90),
                        ingestion, retrain, intervals, explanations, /model/status. The
                        API's model layer.
  marketing/
    copy.py             OfferService: Egyptian-Arabic offer copy (LLM + template fallback +
                        _validate). build() for SKUs, build_freeform() for arbitrary products.
    publisher.py        MetaPublisher: Graph API, dry-run by default, 3 gates for live post.
  integration/
    restomind.py        BRIDGE to the client's system: maps RestoMind product/restaurant
                        shapes -> model output. production_plan(), surplus_offers(),
                        predict_week() (weekly, matches their `predictions` doc).
    registry.py         MULTI-TENANT per-restaurant state. Stores each restaurant's sales
                        separately and learns real demand LEVEL per product from ingested
                        data, so seeding sales actually moves predictions (proven by test).
  api/
    schemas.py          All Pydantic request/response models.
    main.py             FastAPI app, 16 route decorators, CORS, lifespan (trains on startup).
scripts/
    run_backtest.py     Model comparison (all models, both horizons).
    run_simulation.py   Business simulation -> EGP saved vs the manager baseline.
dashboard.py            Streamlit investor demo (4 tabs).
tests/                  79 tests across 7 files (see §6).
data/
    synthetic_pos.parquet   generated dataset
    market_priors.json      LLM-generated calendar sensitivities (the "analysis result")
    models/                 pickled models (if saved)
test/                   CLONED CLIENT REPOS (RestoMindAPI backend, restomind-app frontend)
postman_collection.json 12 requests w/ Arabic docs + auto-tests (import into Postman/newman)
problem_analysis.html   Arabic RTL problem-size pitch page (published as an Artifact)
README.md               project readme
AI_ML_PLAN.md           the approved plan
AI_ARCHITECTURE_ANALYSIS.md   (analysis doc)
```

---

## 4. The model — key technical decisions (READ before touching models)

**Production model = `CalendarDecomposed`** (in `forecaster.py`). It is a decomposition:
`demand = baseline level (LightGBM) × calendar multiplier (Ridge)`.

Decisions, each learned the hard way — do not "simplify" them away:

1. **Decompose calendar from autoregression.** A single LightGBM given both lag AND
   calendar features learns Ramadan almost entirely from `lag_1`/`roll_mean_7`, so it
   REACTS to Ramadan days late (forecast croissants UP on day 1 of Ramadan when true
   demand halves). Adding more years did NOT fix it — structural. Fix: fit calendar
   effects separately (Ridge, calendar-only, log space) so the effect lands on day one,
   and LightGBM models the deseasonalised level. Full write-up in `seasonality.py` header.
2. **Forecast a quantile, not the mean (newsvendor).** `q* = Cu/(Cu+Co)` per item, from
   margin + goodwill vs spoilage-adjusted cost. Waste costs more than a lost sale for
   short-shelf-life staples, so optimal production is often BELOW mean. `items.py`.
3. **Censored demand.** Sell-outs understate demand. `service`/`forecaster` handle it by
   imputation (fit on uncensored, fill censored days with an upper-quantile prediction).
   Measured: dropping stockout rows is WORSE than keeping them.
4. **Event-free baseline for multiplier fitting** (`clean_baseline` in `features.py`):
   using the ordinary 28-day rolling mean underestimates Ramadan effects (the window
   fills with Ramadan days). Fixing this recovered konafa's true 4.5× (was 2.0×).
5. **Metrics:** WAPE headline (MAPE is unusable — items hit zero; it also punishes
   over-forecasting asymmetrically). MASE vs seasonal naive. Pinball (matches quantile).

**Measured performance (SIMULATED, rolling-origin backtest):**
- WAPE **0.162** vs seasonal-naive **0.224** → **~28% better**. MASE < 1.
- Business sim: waste **21%→8%**, **~39,000 EGP/month/branch**, fill rate **~92%**.
- Newsvendor quantile beats plain median by ~108k EGP over the test window.

---

## 5. Cold-start → trained hybrid (the "rule-based then train" flow the owner asked for)

- **Rule-based (day 0):** every item forecasts from `market_priors.json` (calendar
  sensitivities by category) × owner's `avgDailySales`. Knows Ramadan/Eid with no data.
- **Ingest:** backend POSTs nightly actuals to `POST /data/ingest`. `ForecastService`
  counts days per item and refreshes rule baselines immediately.
- **Switch at 90 days/item** (`TRAIN_THRESHOLD_DAYS`, configurable): item promotes to the
  trained `CalendarDecomposed` model automatically.
- **Event-aware safety net:** even a trained item keeps using the RULES for a major event
  (Ramadan/Eid/kahk/Sham) it has NOT yet seen in its training window — so switching never
  makes a holiday forecast worse. It takes over that event only after living through it.
- `GET /model/status` shows each item's mode + progress. `service.py` methods: `train`,
  `start_cold`, `ingest`, `_use_ml(sku, date)`, `status`.

---

## 6. Endpoints (16 routes in `app/api/main.py`)

All routes require `X-API-Key` except `GET /health`, once `REQUIRE_API_KEY`/`API_KEY_HASH`
are set (see `docs/01-api-key-hardening.md`; unset in dev, so this is a no-op locally).
Every route except `/health` also sits behind an always-on cost-guard rate limit (429 +
`Retry-After` past the ceiling; `docs/03-cors-and-rate-limiting.md`) -- tighter on
`/marketing/*` than everything else, and not a real per-user limiter (there's only one
caller, the backend).

Core: `GET /health`, `GET /model/status`, `POST /data/ingest`
Forecasting: `POST /forecast/daily`, `/forecast/weekly`, `/forecast/daily-batch`
  (all items, 1 call, ~6× faster), `/forecast/weekly-batch`, `/forecast/seasonality-adjustment`
Alerts/surplus/marketing: `POST /alerts/waste-prevention`, `/surplus/detect`,
  `/marketing/generate-offer`, `/marketing/publish` (dry-run by default)
**RestoMind bridge:** `POST /integration/restomind/production-plan` (Admin screen),
  `/integration/restomind/surplus-offers` (Stores screen),
  `/integration/restomind/predict` (weekly, maps to their `predictions` collection — routes
  through the per-restaurant registry, so it uses a learned level when data exists),
  `/integration/restomind/ingest` (per-restaurant sales → learns real levels),
  `GET /integration/restomind/status/{restaurant_id}` (which products use learned vs estimate)

Every forecast response carries `confidence`, `source` (rule_based|batch), interval, and
`factors` (calendar attribution) — the explanation is what makes managers trust it.

Tests (**83 total**): `test_egypt_calendar` (calendar dates vs known values), `test_generate`
(effect recovery), `test_forecaster` (decomposition anticipates Ramadan, beats naive),
`test_hybrid` (cold-start→trained + event-aware routing), `test_market_priors`,
`test_api` (all endpoints), `test_restomind_bridge` (the bridge), `test_registry`
(seeding sales changes predictions + per-tenant isolation). Postman: 17 requests,
run via `npx newman run postman_collection.json --env-var "baseUrl=http://127.0.0.1:PORT"`.

---

## 7. RestoMind integration (the client's system)

Cloned into `test/RestoMindAPI` (backend) and `test/restomind-app` (frontend).
**Stack: NestJS + MongoDB (Mongoose).** Studied from its `plan.md` + models. Key facts:

- **Multi-tenant by `restaurantId`.** Products are per-restaurant (unique `restaurantId+title`).
  → each restaurant needs its OWN model/baselines. Our model is currently SINGLE-tenant;
  the bridge is stateless per-request so it works, but a real multi-tenant model registry
  is still TODO.
- **Data models that matter:** `Product` (has `freshnessWindow` = shelf life, `category`,
  `isDeleted`, `isAvailable`), `Order`/`OrderItem` (sales, every item references an `Offer`),
  `Recipe` (product → ingredients bill-of-materials), `Ingredient` (has `shelfLifeDays`,
  stock), `Offer` (has `estimatedWasteReduction`, `actualUnitsSold`, `source: manual|
  ai_recommendation`, `recommendationId`).
- **Sales source = `sales_transactions`** (planned Phase 2): fed by BOTH marketplace orders
  AND bulk POS import; marks `promotionActive`. NOT built yet.
- **`predictions` collection (planned Phase 5)** = `restaurantId, productId, modelVersionId,
  targetWeek, predictedOrders, featuresUsed, actualOrders?`. **THIS IS WHERE OUR MODEL'S
  OUTPUT GOES.** Their backend calls "the AI service" (= us) with computed features and
  stores our `predictedOrders`. `/integration/restomind/predict` already returns this shape.
- **CRITICAL nuance:** their planned feature list (Phase 5) is autoregressive only
  (lags/rolling/promo) with **NO calendar features** — the exact weakness we solved. Our
  bridge ADDS the Egyptian calendar from `targetWeek`. Recommended integration: backend
  sends raw sales via `/data/ingest` + asks predictions by date, and OUR model computes
  calendar itself (preserves the 28% edge). Their weekly `targetWeek` vs our daily is
  reconciled by summing 7 days in `predict_week()`.
- Their AI pipeline is Phases 5/6/8: predictions → waste-reports/recommendations → offers
  → reconcile actuals & accuracy. Bridge is designed to slot into this.

**LIVE INTEGRATION DONE (see `LIVE_DEMO.md`):** their NestJS backend boots on local
MongoDB; `app/integration/seed_restomind.py` seeds a restaurant+products into their Mongo;
`app/integration/connect_restomind.py` reads those products, calls our model over HTTP, and
writes results into their `predictions` collection in the correct schema shape. Registry
state persists to `data/registry_state.pkl` via `REGISTRY_STORE` env var (survives restart).
**Closest-to-real simulation (`seed_bakery_history.py`):** seeds the same 11 items + 2 years
of daily sales the model trained on into their Mongo (products carry `sku`, + 8,019
`sales_transactions`). The connector routes SKU-linked products to the TRAINED model
(`/forecast/weekly`), so predictions in their DB show correct calendar behaviour
(croissant Ramadan 2113→1025, konafa 442→1108), modelVersionId `calendar_decomposed/batch`.
**Still TODO:** their Phase-5 `predictions` module + `sales_transactions` writer (not built by
them yet — connector talks to Mongo directly meanwhile); trained-model for ARBITRARY
per-restaurant products (currently trained path only works for the 11 known SKUs).

---

## 8. Market priors — dynamic per restaurant (`market_priors.json` + `market_priors.py`)

Calendar sensitivities are **category-level** (transferable: all desserts spike in Ramadan)
+ **item overrides** (konafa is THE Ramadan sweet). The FILE is the "LLM analysis result" —
editing it changes forecasts immediately (verified: croissant Ramadan 146→58 when tweaked).
`generate_priors_llm()` can regenerate from a real menu via a free-tier LLM, with validation
+ fallback. Levels (units/day) come from onboarding, NOT the LLM. Everything self-corrects
from data after 90 days.

**Location-based dynamic (discussed, NOT built):** country-level calendar (weekend days +
national holidays via a `holidays`-style lib + existing Hijri) is the big lever and is
moderate effort. City/weather = optional. Hyper-local = learned from data automatically.

---

## 9. What's next / open items (priority order)

1. **Pilot design** — the real proof. Log daily: model forecast vs what the bakery made vs
   actual waste. This is what turns "simulated" into "measured". (User asked for this.)
2. **Live RestoMind wiring** — boot their NestJS + MongoDB, connect to this service, test
   end-to-end. Needs their env/DB. Optionally build a connector that reads
   `sales_transactions` and feeds `/data/ingest` so seeding real data actually moves the model.
3. **Multi-tenant model registry** — PARTIALLY DONE: `integration/registry.py` now keys
   state by `restaurantId` and learns the per-product demand LEVEL from ingested sales
   (rule-based calendar shape still). Remaining: wire the full trained `CalendarDecomposed`
   model per restaurant (needs per-product economics generalised off the built-in catalogue,
   and persistence — it's in-memory now).
4. **Ingredient-level forecasting** — use `Recipe` (bill of materials) to convert product
   demand → ingredient purchasing forecast; use `freshnessWindow`/`shelfLifeDays` instead of
   hardcoded shelf life. Unlocks à-la-carte restaurants (waste is at ingredient level).
5. **Country/location profiles** for the calendar (see §8).
6. **French bakery dataset** (Kaggle `matthieugimbert/french-bakery-daily-sales`, ~2 yrs) —
   optional real-data sanity check of the pipeline (no Egyptian signal, so mechanics only).
   User must download it (Kaggle needs login); then build an adapter to backtest on it.
7. Add the RestoMind bridge requests + `/data/ingest` + `/model/status` to Postman collection.

---

## 10. Working style notes (from the owner)

- Owner writes in Egyptian Arabic; respond in Arabic. Product copy = Egyptian dialect.
- Owner said "always accept / keep building" earlier, but STILL confirm before
  outward-facing/irreversible actions (real social posting, spending money).
- Be an honest technical partner: flag real risks (adoption, "beats the baker's gut?",
  simulated≠measured) rather than cheerleading. The owner values this.
- Global git rule: commit messages must have NO `Co-Authored-By`/attribution trailer.
- Don't auto-run servers unprompted; the owner prefers to run/test themselves — give exact
  commands.
```
