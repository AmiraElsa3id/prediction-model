"""Seed RestoMind's MongoDB with the FULL synthetic bakery the model trained on.

This is the closest-to-real simulation available before a real bakery exists: it puts the
same 11 items and 2 years of daily sales that the trained model learned from into their
Mongo, so their database looks like an established Egyptian bakery with real history.

Each product carries its `sku` (e.g. PASTRY_CROISSANT), so the connector can route it to
the TRAINED CalendarDecomposed model (`/forecast/weekly`) rather than the cold-start rules
-- giving proper Ramadan/Eid-aware predictions (WAPE ~0.16), not just category priors.

Run:  MONGO_URL=mongodb://127.0.0.1:27017/restomind \
      .venv/bin/python -m app.integration.seed_bakery_history
"""

from __future__ import annotations

import datetime as dt
import os

from bson import ObjectId
from pymongo import MongoClient

from app.core.generate import generate
from app.core.items import CATALOGUE

MONGO_URL = os.getenv("MONGO_URL", "mongodb://127.0.0.1:27017/restomind")
RESTAURANT_NAME = "مخبز المحاكاة الكاملة"

# RestoMind category name per our internal category (Arabic, so map_category resolves it).
CATEGORY_AR = {
    "bread": "مخبوزات", "pastry": "معجنات", "cake": "كيك", "sweet": "حلويات شرقية",
    "savoury": "مالح", "seasonal": "موسمي", "dry": "بيسكوت",
}


def seed(mongo_url: str = MONGO_URL) -> dict:
    db = MongoClient(mongo_url).get_default_database()
    now = dt.datetime.now(dt.timezone.utc)

    # Fresh restaurant for the full-history simulation.
    db.restaurants.delete_many({"name": RESTAURANT_NAME})
    restaurant_id = db.restaurants.insert_one({
        "name": RESTAURANT_NAME, "ownerUserId": ObjectId(),
        "address": {"city": "Cairo", "country": "Egypt"},
        "isActive": True, "isDeleted": False, "createdAt": now, "updatedAt": now,
    }).inserted_id

    # Categories + products (one per catalogue item), keyed by sku.
    sku_to_pid: dict[str, ObjectId] = {}
    cat_ids: dict[str, ObjectId] = {}
    for item in CATALOGUE:
        cat_ar = CATEGORY_AR.get(item.category, item.category)
        if cat_ar not in cat_ids:
            existing = db.categories.find_one({"name": cat_ar})
            cat_ids[cat_ar] = existing["_id"] if existing else db.categories.insert_one(
                {"name": cat_ar, "createdAt": now, "updatedAt": now}
            ).inserted_id

        pid = ObjectId()
        sku_to_pid[item.sku] = pid
        db.products.replace_one(
            {"restaurantId": restaurant_id, "title": item.name_ar},
            {
                "_id": pid, "title": item.name_ar, "slug": f"{item.sku.lower()}-{str(pid)[-6:]}",
                "description": item.name_en, "longDescription": item.name_en,
                "price": item.unit_price, "sku": item.sku,   # <- link to the trained model
                "image": {"public_id": "demo", "secure_url": "https://example.com/demo.jpg"},
                "category": cat_ids[cat_ar], "restaurantId": restaurant_id,
                "freshnessWindow": item.shelf_life_days, "isAvailable": True,
                "isDeleted": False, "createdAt": now, "updatedAt": now,
            },
            upsert=True,
        )

    # 2 years of daily sales -> sales_transactions (the history the model learned from).
    data = generate()
    db.sales_transactions.delete_many({"restaurantId": restaurant_id})
    rows = [
        {
            "restaurantId": restaurant_id, "productId": sku_to_pid[r["sku"]], "sku": r["sku"],
            "date": r["date"].to_pydatetime(), "quantity": int(r["sales_qty"]),
            "source": "seed_synthetic", "promotionActive": False,
        }
        for r in data.to_dict("records")
    ]
    for i in range(0, len(rows), 2000):
        db.sales_transactions.insert_many(rows[i:i + 2000])

    return {
        "restaurantId": str(restaurant_id),
        "products": len(sku_to_pid),
        "salesTransactions": len(rows),
        "dateRange": f"{data['date'].min().date()} .. {data['date'].max().date()}",
    }


if __name__ == "__main__":
    out = seed()
    print("SIMULATED full-history bakery seeded into RestoMind Mongo:")
    print(f"  restaurantId      : {out['restaurantId']}")
    print(f"  products          : {out['products']} (each linked to a trained SKU)")
    print(f"  sales_transactions: {out['salesTransactions']:,}  ({out['dateRange']})")
