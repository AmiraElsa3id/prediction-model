"""Live connector: RestoMind MongoDB  ->  our model  ->  RestoMind `predictions`.

This is what RestoMind's Phase-5 AI pipeline will do, realised as a standalone script so
we can demonstrate the full live loop before their pipeline is built:

  1. read `restaurants` + `products` from their MongoDB
  2. for each product, call our model over HTTP (the same service their backend would call)
  3. write the result into a `predictions` collection matching their prediction schema
     (restaurantId, productId, modelVersionId, targetWeek, predictedOrders, featuresUsed)

So the model reads their real data structures and writes predictions exactly where their
system expects them -- a genuine service-to-service integration, not a mock.

Run (with Mongo + the model API both up):
  MONGO_URL=mongodb://127.0.0.1:27017/restomind \
  MODEL_URL=http://127.0.0.1:8200 \
  .venv/bin/python -m app.integration.connect_restomind 2025-03-10
"""

from __future__ import annotations

import datetime as dt
import os
import sys

import httpx
from pymongo import MongoClient

MONGO_URL = os.getenv("MONGO_URL", "mongodb://127.0.0.1:27017/restomind")
MODEL_URL = os.getenv("MODEL_URL", "http://127.0.0.1:8200")


def _category_name(db, category_id) -> str | None:
    if not category_id:
        return None
    cat = db.categories.find_one({"_id": category_id})
    return cat.get("name") if cat else None


def _predict_trained(model_url, rid, product, sku, target_week) -> dict:
    """Weekly forecast from the TRAINED model, normalised to the prediction doc shape."""
    r = httpx.post(f"{model_url}/forecast/weekly",
                   json={"sku": sku, "start_date": target_week.isoformat()}, timeout=20.0)
    r.raise_for_status()
    w = r.json()
    days = w["days"]
    # Strongest distinct calendar driver across the week.
    seen: dict[str, dict] = {}
    for d in days:
        for f in d.get("factors", []):
            if f["factor"] not in seen or abs(f["impact_pct"]) > abs(seen[f["factor"]]["impact_pct"]):
                seen[f["factor"]] = f
    factors = sorted(seen.values(), key=lambda f: abs(f["impact_pct"]), reverse=True)
    source = days[0]["source"] if days else "batch"
    return {
        "modelVersionId": f"calendar_decomposed/{source}",
        "targetWeek": target_week.isoformat(),
        "predictedOrders": w["total_quantity"],
        "featuresUsed": {"sku": sku, "mode": source, "model": "calendar_decomposed",
                         "confidence": days[0].get("confidence") if days else None},
        "factors": factors,
    }


def _predict_no_sku(model_url, rid, product, category, target_week) -> dict:
    """Prediction via the bridge for a product with no trained SKU link.

    The bridge reports the owner-estimated or learned level; a product with neither
    comes back flagged as still training rather than given a guessed number.
    """
    r = httpx.post(f"{model_url}/integration/restomind/predict",
                   json={"restaurantId": rid, "productId": str(product["_id"]),
                         "title": product["title"], "category": category,
                         "targetWeek": target_week.isoformat(),
                         "avgDailySales": product.get("_avgDailySales", 60)}, timeout=15.0)
    r.raise_for_status()
    return r.json()


def run(target_week: dt.date, mongo_url: str = MONGO_URL, model_url: str = MODEL_URL) -> list[dict]:
    db = MongoClient(mongo_url).get_default_database()
    written: list[dict] = []

    for restaurant in db.restaurants.find({"isDeleted": {"$ne": True}}):
        rid = str(restaurant["_id"])
        products = db.products.find({"restaurantId": restaurant["_id"], "isDeleted": {"$ne": True}})

        for p in products:
            sku = p.get("sku")
            if sku:
                # Product is linked to a trained SKU -> use the TRAINED model (calendar-
                # decomposed, Ramadan/Eid-aware) via the weekly forecast endpoint.
                pred = _predict_trained(model_url, rid, p, sku, target_week)
            else:
                # No trained SKU -> bridge (owner estimate or learned level).
                pred = _predict_no_sku(model_url, rid, p, _category_name(db, p.get("category")),
                                       target_week)

            # Write into their predictions collection, in their document shape.
            doc = {
                "restaurantId": restaurant["_id"],
                "productId": p["_id"],
                "modelVersionId": pred["modelVersionId"],
                "targetWeek": pred["targetWeek"],
                "predictedOrders": pred["predictedOrders"],
                "featuresUsed": pred["featuresUsed"],
                "factors": pred["factors"],
                "actualOrders": None,
                "createdAt": dt.datetime.now(dt.timezone.utc),
            }
            db.predictions.replace_one(
                {"restaurantId": restaurant["_id"], "productId": p["_id"],
                 "targetWeek": pred["targetWeek"]},
                doc, upsert=True,
            )
            written.append({"title": p["title"], "predictedOrders": pred["predictedOrders"],
                            "topFactor": pred["factors"][0]["factor"] if pred["factors"] else "-"})

    return written


if __name__ == "__main__":
    week = dt.date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else dt.date.today()
    rows = run(week)
    print(f"wrote {len(rows)} predictions -> `predictions` collection (targetWeek {week})")
    for r in rows:
        print(f"  {r['title']:16s} predictedOrders={r['predictedOrders']:5d}  ({r['topFactor']})")
