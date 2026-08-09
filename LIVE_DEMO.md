# Live Integration Demo — RestoMind ↔ Model ↔ predictions

Proven working end-to-end: the model reads RestoMind's real MongoDB data and writes
predictions back where their system expects them. This is a genuine service-to-service
loop, not a mock. All figures are still SIMULATED (no real sales history).

## What runs

| Piece | Command | Port |
|---|---|---|
| MongoDB | `mongod --dbpath <dir> --port 27017` | 27017 |
| RestoMind backend (NestJS) | `cd test/RestoMindAPI && npm run start` | 3000 |
| Our model API | `REGISTRY_STORE=data/registry_state.pkl .venv/bin/uvicorn app.api.main:app --port 8200` | 8200 |

`test/RestoMindAPI/.env` is set to `DB_URL=mongodb://127.0.0.1:27017/restomind`, `PORT=3000`.

## The loop (run in order)

```bash
# 1. Seed a demo restaurant + products straight into their Mongo (bypasses auth/Cloudinary)
MONGO_URL=mongodb://127.0.0.1:27017/restomind \
  .venv/bin/python -m app.integration.seed_restomind

# 2. Connector: read products from their Mongo -> call our model over HTTP -> write predictions
MONGO_URL=mongodb://127.0.0.1:27017/restomind MODEL_URL=http://127.0.0.1:8200 \
  .venv/bin/python -m app.integration.connect_restomind 2025-03-10

# 3. Confirm predictions landed in THEIR Mongo, in their schema shape
mongosh --port 27017 --quiet restomind --eval 'db.predictions.find().pretty()'
```

Verified output: 5 predictions written, e.g. `كنافة` highest in a Ramadan week, each doc
carrying `restaurantId, productId, modelVersionId, targetWeek, predictedOrders,
featuresUsed{calendar…}, factors, actualOrders:null` — a drop-in for their `predictions`
collection.

## Closest-to-real simulation: seed the FULL trained bakery

Instead of the 5-product cold-start demo, seed the same 11 items + 2 years of daily sales
that the model actually trained on, so their DB looks like an established bakery and
predictions come from the TRAINED model (not the cold-start "still training" state):

```bash
# Seed 11 products (each linked to a trained SKU) + 8,019 sales_transactions into their Mongo
MONGO_URL=mongodb://127.0.0.1:27017/restomind \
  .venv/bin/python -m app.integration.seed_bakery_history

# Connector routes SKU-linked products to the TRAINED model (/forecast/weekly)
MONGO_URL=mongodb://127.0.0.1:27017/restomind MODEL_URL=http://127.0.0.1:8200 \
  .venv/bin/python -m app.integration.connect_restomind 2025-03-10   # Ramadan week
```

Verified trained-model output stored in their `predictions` (modelVersionId
`calendar_decomposed/batch`), showing correct Egyptian calendar behaviour on their data:

| product | normal week | Ramadan week |
|---|---|---|
| كرواسون (croissant) | 2113 | **1025** ↓ (breakfast falls ~52%) |
| كنافة (konafa) | 442 | **1108** ↑ (Ramadan sweet ×2.5) |
| بسبوسة (basbousa) | 1111 | **2151** ↑ (×2) |

`connect_restomind.py` picks the path per product: `product.sku` present → trained
`/forecast/weekly`; otherwise → the bridge `/integration/restomind/predict` (basis level,
or "still training" when there is no basis).

## Persistence (survives restart)

The model API was started with `REGISTRY_STORE=data/registry_state.pkl`. Ingesting sales
writes learned levels to that file; a fresh process reloads them:
```bash
# ingest sales -> data/registry_state.pkl is written; a new RestaurantRegistry(persist_path=...)
# reloads the learned level (verified: konafa level 31.3 survived a simulated restart).
```
Without `REGISTRY_STORE` the registry stays in-memory (tests use this, so they're isolated).

## Files (in app/integration/)

- `seed_restomind.py` — insert restaurant + products into their Mongo (local testing only).
- `connect_restomind.py` — the live connector (Mongo → model HTTP → predictions). This is
  what their Phase-5 pipeline will do; standalone for now so we can demo before it's built.
- `registry.py` — per-restaurant learned levels, now file-persistable.

## Honest status

- ✅ Their NestJS backend boots and serves; MongoDB running; real collections.
- ✅ Model reads their product shapes and writes their prediction shape, over HTTP.
- ✅ Persistence across restart.
- 🟡 The connector talks to Mongo directly (their Phase-5 `predictions` module isn't built
  yet). When they build it, it calls the same model endpoint — no change our side.
- 🟡 Predictions are basis-level only (no real sales history). Trained-model-per-
  restaurant is the next step (HANDOFF §9).
- Numbers are SIMULATED, not measured. Real proof still needs a pilot.

## Stop everything

```bash
pkill -f "uvicorn app.api.main"      # model API
pkill -f "nest start|dist/main"      # RestoMind backend
pkill -f "mongod --dbpath"           # MongoDB
```
