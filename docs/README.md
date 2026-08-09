# Documentation

Complete guides, architecture, and implementation plans.

## Getting Started

| # | Topic | Purpose |
|---|-------|---------|
| **[00-handoff.md](00-handoff.md)** | Complete state dump | Full context: what this is, how to run it, the model architecture, deployment notes. Read this first if you're new. |
| **[01-getting-started.md](01-getting-started.md)** | Quick setup & run | Environment, installation, running the service/tests/backtest/simulation, configuration. |

## Architecture & Integration

| # | Topic | Purpose |
|---|-------|---------|
| **[03-architecture-analysis.md](03-architecture-analysis.md)** | How the model works | Deep dive: the two core ideas, layer-by-layer flow, technical decisions, why certain things work the way they do. |
| **[02-integration-guide.md](02-integration-guide.md)** | RestoMind bridge | Connecting the model to the RestoMind backend: the multi-tenant registry, ingestion, production plans, weekly predictions, surplus detection. |
| **[05-demo.md](05-demo.md)** | Live demonstration | How to run the Streamlit dashboard and Postman API tests for stakeholders. |

## Security & Operations (Implementation Plans)

Detailed plans for hardening the API and integrating with the backend's auth layer.

| # | Topic | Status |
|---|-------|--------|
| **[06-api-key-hardening.md](06-api-key-hardening.md)** | All-endpoint API key auth | ✅ Implemented |
| **[07-backend-api-key-integration.md](07-backend-api-key-integration.md)** | Backend team changes | Handoff doc for RestoMind (not this repo's work) |
| **[08-cors-and-rate-limiting.md](08-cors-and-rate-limiting.md)** | Rate limiting & CORS | ✅ Implemented |

---

## File organization

```
docs/
  00-handoff.md                       Complete state dump for onboarding
  01-getting-started.md               Setup, run, config
  02-integration-guide.md             RestoMind bridge & multi-tenant registry
  03-architecture-analysis.md         Model design, technical decisions
  05-demo.md                          Dashboard & API demo
  06-api-key-hardening.md             API key auth plan (implemented)
  07-backend-api-key-integration.md   Backend team handoff
  08-cors-and-rate-limiting.md        Rate limit plan (implemented)
  README.md                           This index
```

## Quick links

- **Root `README.md`** — project overview & headline results
- **Root `AI_ML_PLAN.md`** — original project plan
- **Root `postman_collection.json`** — 12 API endpoints with auto-tests
- **Root `problem_analysis.html`** — problem-size pitch (Arabic)
- **`app/` folder** — source code (core, models, marketing, integration, api)
- **`scripts/` folder** — backtest, simulation, dashboard
- **`tests/` folder** — 119 unit & integration tests

---

*Last updated: 2026-08-09. All results are from simulated data (pre-launch bakery). Real proof requires a pilot.*
