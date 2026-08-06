# Bakery Demand & Surplus AI — POC

An AI layer for Egyptian bakeries: forecast production to cut waste, and automatically
market end-of-day surplus. Built as a **proof of concept for stakeholders**, on top of an
existing e-commerce/POS system.

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
| Calendar effect for a date (Ramadan/Eid/…), with attribution | `POST /forecast/seasonality-adjustment` |
| Warn when a manual production entry will cause waste | `POST /alerts/waste-prevention` |
| Detect stagnant stock near closing and size a discount | `POST /surplus/detect` |
| Write Egyptian-Arabic promo copy | `POST /marketing/generate-offer` |
| Publish to Facebook/Instagram (dry-run by default) | `POST /marketing/publish` |
| **Post end-of-day actuals so the model learns** | `POST /data/ingest` |
| **Per-item mode: rule-based vs trained model** | `GET /model/status` |

## Headline results (simulated, rolling-origin backtest)

- **Forecast accuracy:** WAPE **0.162** vs seasonal-naive **0.224** — a **28% improvement**,
  and the model *anticipates* Ramadan/Eid rather than lagging them.
- **Business impact:** production cost (waste + lost sales incl. goodwill) down **~19%**,
  ≈ **39,000 EGP/month** for one branch. Waste rate **21% → 8%**, while fill rate stays at
  **~92%** of demand served.

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

## Cold start → trained model (hybrid)

A brand-new bakery has no history, so the system does **not** wait for data to be useful:

- **Rule-based (day 0+):** each item forecasts from owner priors + hand-encoded Egyptian
  calendar rules. It already knows croissants fall in Ramadan and kahk only sells before
  Eid — no training required.
- **Trained model (per item, at 90 days):** the backend posts each night's actuals to
  `POST /data/ingest`. Once an item reaches **90 days** of history it switches automatically
  to the trained `CalendarDecomposed` model. `GET /model/status` shows each item's mode and
  progress.
- **Event-aware safety net:** even a "trained" item keeps using the rules for a major event
  (Ramadan, Eid, kahk season, Sham El-Nessim) it has **not yet lived through** — so switching
  to the model never makes a holiday forecast *worse*. It takes over that event only after it
  has real data for it.

The 90-day threshold is configurable (`ForecastService(train_threshold=...)`). Start the
service in cold-start mode with `COLD_START=true`.

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
    forecaster.py       SeasonalNaive, LightGBMQuantile, CalendarDecomposed (production)
    rule_based.py       cold-start forecaster: owner priors + calendar rules, no training
    service.py          hybrid routing, ingestion, intervals, explanations — model layer
  marketing/
    copy.py             Egyptian-Arabic copy (LLM + template fallback, validated)
    publisher.py        Meta Graph API client (dry-run by default)
  api/
    schemas.py, main.py FastAPI service, Pydantic I/O, Swagger
scripts/
    run_backtest.py     model comparison
    run_simulation.py   the EGP-saved business simulation
dashboard.py            Streamlit investor demo
tests/                  67 tests: calendar dates, effect recovery, API, cold-start, guards
postman_collection.json 12 endpoints with Arabic docs + auto-tests (import into Postman)
```

## Run it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

.venv/bin/python -m app.core.generate      # regenerate the synthetic dataset
.venv/bin/python -m pytest tests/ -q       # 57 tests
.venv/bin/python -m scripts.run_backtest   # model comparison table
.venv/bin/python -m scripts.run_simulation # the money slide

.venv/bin/uvicorn app.api.main:app --reload   # API + Swagger at /docs
.venv/bin/streamlit run dashboard.py          # investor dashboard
```

## Configuration (optional, for the live-service features)

| Variable | Purpose |
|---|---|
| `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL` | free-tier LLM for Arabic copy (Groq/Gemini/OpenRouter). Without it, templates are used. |
| `META_PAGE_ID`, `META_ACCESS_TOKEN`, `META_PUBLISH_ENABLED` | live Meta publishing. All three required; otherwise `/marketing/publish` returns a preview. |
| `COLD_START` | `true` starts with zero history (fully rule-based) for the cold-start demo. Default trains on the full simulated dataset. |
| `CORS_ORIGINS` | comma-separated frontend origins allowed to call the API. Default `http://localhost:3000`; set explicitly for other deployments. |

## What's needed to move beyond the POC

The forecast is only as good as its inputs. Before a real deployment we need: confirmation
the inventory schema exposes closing stock + stock-added per item/day; real shelf life and
unit economics per item (these set `q*`); branch/SKU counts and opening hours; and
eventually 12+ months of real POS history, at which point the cold-start priors hand over
to the batch models.
