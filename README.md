# Bakery Demand & Surplus AI — POC

An AI layer for Egyptian bakeries: forecast production to cut waste and detect surplus
that needs quick discounting. Built as a **proof of concept for stakeholders**, on top of
an existing e-commerce/POS system.

> ### ⚠️ All results here are SIMULATED
> The bakery is pre-launch, so there is no real sales history yet, and no public dataset
> carries Egyptian calendar signal (Ramadan, Eid, Coptic feasts, Fri/Sat weekend). The
> system is therefore built and evaluated on a **synthetic dataset generated with known
> effect sizes**. Every figure below is a projection of what the system would achieve once
> connected to a real POS — not a measurement from a real bakery. This is stated on the
> API landing page, the `/health` endpoint, and every chart in the dashboard.

---

## What it does

| Capability | Endpoint |
|---|---|
| Daily production forecast, with confidence interval and explanation | `POST /forecast/daily` |
| **Daily forecast for every item in one call** (for the POS backend) | `POST /forecast/daily-batch` |
| 7-day production plan (single item / all items) | `POST /forecast/weekly`, `POST /forecast/weekly-batch` |
| **Register/refresh item economics (dynamic catalogue)** | `POST /catalogue/upsert` |
| Calendar effect for a date (Ramadan/Eid/…), with attribution | `POST /forecast/seasonality-adjustment` |
| Warn when a manual production entry will cause waste | `POST /alerts/waste-prevention` |
| Detect stagnant stock near closing and size a discount | `POST /surplus/detect` |
| **Post end-of-day actuals so the model learns** | `POST /data/ingest` |
| **Per-item status: untrained vs trained model** | `GET /model/status` |

## Headline results (simulated, rolling-origin backtest)

- **Forecast accuracy:** WAPE **0.162** vs seasonal-naive **0.224** — a **28% improvement**,
  and the model *anticipates* Ramadan/Eid rather than lagging them.
- **Business impact:** production cost (waste + lost sales incl. goodwill) down **~19%**,
  ≈ **39,000 EGP/month** for one branch. Waste rate **21% → 8%**, while fill rate stays at
  **~92%** of demand served.

The EGP figure is not a constant in the code — it is the printed output of one
`python -m scripts.run_simulation` run against the generated dataset (best policy vs the
simulated manager's own production, over 6 rolling-origin folds × 28 test days, scaled to
30 days). Regenerating the data or changing the folds moves it. Reproduce with
`python -m app.core.generate && python -m scripts.run_simulation`.

## The two ideas that carry the project

1. **Forecast a profit-optimal quantile, not the mean.** Over-producing wastes full cost;
   under-producing loses only margin (plus goodwill). The newsvendor optimum
   `q* = Cu/(Cu+Co)` is set *per item* from its economics and shelf life. In simulation the
   quantile model beats a plain median model by ~112k EGP over the test window.
2. **Decompose the calendar from the autoregressive level.** A single model given both lag
   and calendar features learns to predict Ramadan almost entirely from yesterday's sales,
   so it reacts to events days late. Splitting the calendar effect (ridge, calendar-only)
   from the level (LightGBM on deseasonalised demand) makes the effect land on day one —
   and yields an *exact* explanation of every forecast. See
   [app/models/seasonality.py](app/models/seasonality.py) for the measurements behind this.

## Cold start → trained model

A brand-new bakery has no history yet: until an item crosses the training threshold there
is **no forecast at all** — the system is honest instead of handing the store a guess.

- **Untrained (before 90 days):** `GET /model/status` reports the item `untrained`, and the
  bridge endpoints answer with a `trainingMessage` ("still training") and `0` quantity when
  there is no basis. No priors, no hand-encoded calendar rules are applied.
- **Trained model (per item, at 90 days):** the backend posts each night's actuals to
  `POST /data/ingest`. Once an item reaches **90 days** of history it switches automatically
  to the trained `CalendarDecomposed` model. `GET /model/status` shows each item's mode and
  progress.
- **Unseen-event confidence:** a major event (Ramadan, Eid, kahk season, Sham El-Nessim) the
  model trained but has **not lived through** is still forecast — but flagged `low`
  confidence with a wider interval, never silently upgraded.

The 90-day threshold is configurable (`ForecastService(train_threshold=...)`). Start the
service with no history via `COLD_START=true`.

## Dynamic catalogue (items are what you upload, not a fixed 11)

There is no hardcoded item list at runtime. The service learns its items from uploaded
data and nothing else:

- On a clean start (`COLD_START=true`) `/model/status` reports **zero** items and
  `/forecast/*` answers *not found* until data arrives — it never pretends to know
  products the bakery hasn't given it.
- Items become known in one of two ways: the backend posts its product menu to
  `POST /catalogue/upsert` (SKU + price/cost/shelf life), or sales arrive at
  `POST /data/ingest` carrying those economics — a SKU is registered the first night
  its actuals arrive.
- The newsvendor quantile (how conservatively each item is forecast) is derived **per
  item from its own uploaded economics**, `q* = Cu/(Cu+Co)`, so a brand-new SKU gets a
  sane service level from day one.
- `/model/status`, `/forecast/*`, `/alerts/*`, `/surplus/*` all read
  the same per-service `Catalogue`, so an unknown SKU is a clear 404 with a hint, never
  a guess.

The 11-item set still exists in `app/core/items.py` — as **simulation fixtures** for the
synthetic generator, the backtest, the business simulation and the demo dashboard. The
running service never consults it.

## Documentation

Full guides live in **[`docs/`](docs/)**, organized by topic:

- **[docs/00-handoff.md](docs/00-handoff.md)** — complete state dump for onboarding
- **[docs/01-getting-started.md](docs/01-getting-started.md)** — setup, run, and config
- **[docs/02-integration-guide.md](docs/02-integration-guide.md)** — RestoMind bridge integration
- **[docs/03-architecture-analysis.md](docs/03-architecture-analysis.md)** — model design & technical decisions
- **[docs/05-demo.md](docs/05-demo.md)** — live dashboard & API demo
- **[docs/06-08-*.md](docs/)** — security & implementation plans

See **[docs/README.md](docs/README.md)** for the full index.

## Architecture

```
app/
  core/
    egypt_calendar.py   Ramadan/Eid (Hijri) + Coptic + school/holiday/payday features
    items.py            item catalogue: economics, shelf life, newsvendor q*
    generate.py         synthetic POS+inventory generator (known injected effects)
    features.py         cleaning, censored-demand handling, lag/calendar features
    surplus.py          near-closing surplus detection + discount tiers
    evaluation.py       WAPE/MASE/pinball + rolling-origin backtest
  models/
    seasonality.py      per-item calendar multipliers (ridge, log space)
forecaster.py       SeasonalNaive, MovingAverage, LightGBMQuantile,
                        CalendarDecomposed (production)
    service.py          threshold-gated routing, ingestion, intervals, explanations — model layer
  integration/
    restomind.py        bridge to RestoMind's field shapes; routes SKU-linked products
                        through the trained model, else a learned/owner basis level
    registry.py         multi-tenant per-restaurant state; learns each product's level
    mongo_store.py      durable MongoDB persistence for the registry
    connect_restomind.py live loop: read their Mongo → call the model → write predictions
    seed_restomind.py, seed_bakery_history.py   demo/local seeding
  api/
    schemas.py, main.py FastAPI service, Pydantic I/O, Swagger
scripts/
    run_backtest.py     model comparison
    run_simulation.py   the EGP-saved business simulation
dashboard.py            Streamlit investor demo
tests/                  113 tests: calendar dates, effect recovery, API, cold-start,
                        integration bridge/registry/store, guards
postman_collection.json endpoints with Arabic docs + auto-tests (import into Postman)
```

## Run it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

.venv/bin/python -m app.core.generate      # regenerate the synthetic dataset
.venv/bin/python -m pytest tests/ -q       # 113 tests
.venv/bin/python -m scripts.run_backtest   # model comparison table
.venv/bin/python -m scripts.run_simulation # the money slide

.venv/bin/uvicorn app.api.main:app --reload   # API + Swagger at /docs
.venv/bin/streamlit run dashboard.py          # investor dashboard
```

## Configuration (optional, for the live-service features)

| Variable | Purpose |
|---|---|
| `COLD_START` | `true` starts with zero history (every item untrained, no forecast until it crosses the threshold) for the cold-start demo. Default trains on the full simulated dataset. |
| `REQUIRE_API_KEY`, `API_KEY_HASH` | API key auth for every route except `/health` (see [docs/06-api-key-hardening.md](docs/06-api-key-hardening.md)). Unset in local dev, auth is skipped. `API_KEY_HASH` is the SHA-256 hex digest of the real key, not the key itself — the raw key lives only on the caller's side. Callers send it as `X-API-Key`. Rotation is manual only for now. |

No `CORS_ORIGINS`: this service is never called from a browser directly, only the
backend calls it server-to-server, so there's no CORS layer to configure
([docs/08-cors-and-rate-limiting.md](docs/08-cors-and-rate-limiting.md) §2.1). Every route except `/health` also sits
behind a blunt, always-on cost-guard rate limit (not a real per-user limiter — see
[docs/08](docs/08-cors-and-rate-limiting.md) §1.2 for why); its thresholds are constants in `app/api/ratelimit.py`.

## What's needed to move beyond the POC

The forecast is only as good as its inputs. Before a real deployment we need: confirmation
the inventory schema exposes closing stock + stock-added per item/day; real shelf life and
unit economics per item (these set `q*`); branch/SKU counts and opening hours; and
eventually 12+ months of real POS history, at which point cold-start items hand over to
the trained models once they cross the training threshold.
