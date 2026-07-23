"""Seed a demo restaurant + products directly into RestoMind's MongoDB.

Bypasses the auth + Cloudinary image flow (which the REST API requires) by inserting
documents that match RestoMind's Mongoose schemas. For LOCAL TESTING ONLY -- it lets the
live connector (`connect_restomind.py`) have real products to forecast without a full
onboarding flow.

Run:  MONGO_URL=mongodb://127.0.0.1:27017/restomind .venv/bin/python -m app.integration.seed_restomind
"""

from __future__ import annotations

import datetime as dt
import os

from bson import ObjectId
from pymongo import MongoClient

MONGO_URL = os.getenv("MONGO_URL", "mongodb://127.0.0.1:27017/restomind")

RESTAURANT = {"name": "مخبز الديمو", "city": "Cairo"}

# (title, category-name, price EGP, freshnessWindow days, avgDailySales)
PRODUCTS = [
    ("كرواسون", "معجنات", 18.0, 2, 180),
    ("كنافة", "حلويات شرقية", 45.0, 2, 40),
    ("فطير مشلتت", "مالح", 60.0, 1, 55),
    ("عيش فينو", "مخبوزات", 2.5, 1, 420),
    ("جاتوه", "كيك", 35.0, 2, 95),
]


def _slugify(text: str) -> str:
    return "-".join(text.split()) + "-" + str(ObjectId())[-6:]


def seed(mongo_url: str = MONGO_URL) -> dict:
    db = MongoClient(mongo_url).get_default_database()
    now = dt.datetime.now(dt.timezone.utc)

    # A restaurant (owner is a placeholder id -- fine for read-only forecasting).
    restaurant_id = ObjectId()
    db.restaurants.replace_one(
        {"name": RESTAURANT["name"]},
        {
            "_id": restaurant_id, "name": RESTAURANT["name"], "ownerUserId": ObjectId(),
            "address": {"city": RESTAURANT["city"], "country": "Egypt"},
            "isActive": True, "isDeleted": False, "createdAt": now, "updatedAt": now,
        },
        upsert=True,
    )
    # Re-read in case it already existed with a different _id.
    restaurant_id = db.restaurants.find_one({"name": RESTAURANT["name"]})["_id"]

    # One category per distinct product category name, then the products.
    cat_ids: dict[str, ObjectId] = {}
    product_ids = []
    for title, cat_name, price, freshness, _avg in PRODUCTS:
        if cat_name not in cat_ids:
            existing = db.categories.find_one({"name": cat_name})
            cat_ids[cat_name] = existing["_id"] if existing else db.categories.insert_one(
                {"_id": ObjectId(), "name": cat_name, "createdAt": now, "updatedAt": now}
            ).inserted_id

        pid = ObjectId()
        db.products.replace_one(
            {"restaurantId": restaurant_id, "title": title},
            {
                "_id": pid, "title": title, "slug": _slugify(title),
                "description": title, "longDescription": title, "price": price,
                "rating": 0, "reviewsCount": 0, "isBestseller": False, "isAvailable": True,
                "image": {"public_id": "demo", "secure_url": "https://example.com/demo.jpg"},
                "category": cat_ids[cat_name], "restaurantId": restaurant_id,
                "freshnessWindow": freshness, "tags": [], "isDeleted": False,
                "createdAt": now, "updatedAt": now,
            },
            upsert=True,
        )
        real = db.products.find_one({"restaurantId": restaurant_id, "title": title})
        product_ids.append(str(real["_id"]))

    return {"restaurantId": str(restaurant_id), "productIds": product_ids}


if __name__ == "__main__":
    out = seed()
    print(f"seeded restaurant {out['restaurantId']} with {len(out['productIds'])} products")
    for pid in out["productIds"]:
        print(f"  product {pid}")
