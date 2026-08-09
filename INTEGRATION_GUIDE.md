# دليل ربط الموديل بالفرونت والباك (RestoMind) — بالتفصيل

> **للفريق:** ده الدليل الكامل اللي بيشرح إزاي الباك اند (NestJS) والفرونت اند بتوع RestoMind
> بيتكلموا مع خدمة الذكاء الاصطناعي (الموديل)، وإزاي كلكم تعملوا seed لنفس الداتا عشان
> المشروع يبقى متطابق عند الجميع.
>
> الموديل = خدمة مستقلة (Python/FastAPI) بتشتغل على بورت لوحدها. الباك بينده عليها HTTP.

---

## 1. المعمار — مين بينده مين؟

```
┌──────────────┐        ┌──────────────────┐        ┌────────────────────┐
│  الفرونت اند  │ ─HTTP→ │  الباك اند        │ ─HTTP→ │  خدمة الموديل        │
│ (restomind-app)│       │  (NestJS)         │        │  (FastAPI, بايثون)  │
│               │ ←JSON─ │                   │ ←JSON─ │                    │
└──────────────┘        └────────┬─────────┘        └────────────────────┘
                                 │
                          ┌──────▼──────┐
                          │  MongoDB     │  ← restaurants, products, sales_transactions, predictions
                          └─────────────┘
```

**القاعدة الذهبية:**
- **الفرونت ما بينداش الموديل مباشرة.** الفرونت بينده الباك بس.
- **الباك** هو اللي بينده الموديل، بياخد النتيجة، **يخزّنها في `predictions`**، والفرونت بيقرأها من الباك.
- ليه؟ عشان الأمان (الموديل مايتعرّضش للنت مباشرة)، والباك يقدر يخزّن ويراجع النتايج.

> ملاحظة: الموديل ماعادش فيه CORS خالص — القاعدة الذهبية بقت متفعّلة في الكود مش بس
> اتفاق: الفرونت مايقدرش ينده الموديل مباشرة من المتصفح، لازم يعدّي على الباك
> (`docs/03-cors-and-rate-limiting.md`).

---

## 2. تشغيل الموديل (خطوة الباك اند)

```bash
cd model
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
REGISTRY_STORE=data/registry.json .venv/bin/uvicorn app.api.main:app --port 8200
```
الموديل دلوقتي على `http://127.0.0.1:8200`. حطّوا العنوان ده في env بتاع الباك:
```
AI_SERVICE_URL=http://127.0.0.1:8200
```

---

## 3. الـ Endpoints اللي الباك محتاج ينده عليها

### 3.1 توقّع أسبوعي — لتخزينه في `predictions`

ده اللي بيتغذّى منه مجموعة `predictions` (Phase 5 بتاعتكم).

**Request:**
```http
POST {AI_SERVICE_URL}/integration/restomind/predict
Content-Type: application/json

{
  "restaurantId": "665f...",       // ObjectId كـ string
  "productId":    "665f...",       // ObjectId كـ string
  "title":        "كنافة",
  "category":     "حلويات شرقية",   // اسم الفئة (بالعربي أو الإنجليزي)
  "targetWeek":   "2025-03-10",     // أول يوم في الأسبوع (YYYY-MM-DD)
  "avgDailySales": 40,              // تقدير مبدئي (لحد ما تيجي مبيعات فعلية)
  "promotionActive": false          // فيه عرض/خصم الأسبوع ده؟
}
```

**Response (بيتطابق مع document في مجموعة `predictions`):**
```json
{
  "restaurantId": "665f...",
  "productId": "665f...",
  "modelVersionId": "restomind-bridge/rule_based-v0.1",
  "targetWeek": "2025-03-10",
  "predictedOrders": 750,
  "confidence": "low",
  "featuresUsed": {
    "mode": "rule_based",
    "baseDailyLevel": 40,
    "levelSource": "owner_estimate",
    "categoryResolved": "sweet",
    "calendar": { "isRamadan": true, "daysToEidFitr": 20 }
  },
  "factors": [ { "factor": "Ramadan", "impact_pct": 150, "direction": "increase" } ],
  "dailyBreakdown": [ { "date": "2025-03-10", "qty": 100, "factors": [...] }, ... ]
}
```

**الباك بياخد الرد ده ويخزّنه** في `predictions` كده (المفاتيح متطابقة تقريبًا):
`restaurantId, productId, modelVersionId, targetWeek, predictedOrders, featuresUsed, actualOrders:null`.

### 3.2 خطة إنتاج يومية — لشاشة الأدمن

```http
POST {AI_SERVICE_URL}/integration/restomind/production-plan
{
  "restaurantId": "665f...",
  "date": "2025-03-15",
  "products": [
    { "productId":"p1", "title":"كرواسون", "category":"معجنات",
      "price":18, "freshnessWindow":2, "avgDailySales":180 }
  ]
}
```
**Response:** `{ totalRecommendedQty, items:[{ productId, recommendedQty, lowerBound, upperBound, confidence, source, factors }] }`

### 3.3 فائض + عروض — لشاشة الستورز

```http
POST {AI_SERVICE_URL}/integration/restomind/surplus-offers
{
  "restaurantId": "665f...",
  "timestamp": "2025-03-15T19:30:00",   // قرب القفل
  "closeHour": 22,
  "stock": [
    { "productId":"p1","title":"كرواسون","category":"معجنات",
      "price":18,"freshnessWindow":2,"avgDailySales":180,"currentStock":40 }
  ]
}
```
**Response:** `{ itemsAtRisk:[{ productId, projectedSurplus, riskScore, suggestedDiscountPct, offerCopyAr, newPrice }] }`

### 3.4 إدخال مبيعات — عشان الموديل يتعلّم المستوى الحقيقي

كل ليلة، بعّتوا مبيعات اليوم عشان التوقّع يتحسّن (بدل تقدير `avgDailySales`):
```http
POST {AI_SERVICE_URL}/integration/restomind/ingest
{
  "restaurantId": "665f...",
  "records": [
    { "date":"2025-07-20", "productId":"665f...", "salesQty":112 }
  ],
  "products": [ { "productId":"665f...", "title":"كنافة", "category":"حلويات شرقية" } ]
}
```
بعد كده، أي `/predict` للمطعم ده بيستخدم **المستوى المتعلّم** بدل التقدير.

---

## 4. كود الباك اند (NestJS) — مثال جاهز

أنشئوا موديول `predictions`، وفيه service بينده الموديل:

```typescript
// src/predictions/ai.service.ts
import { HttpService } from '@nestjs/axios';
import { Injectable } from '@nestjs/common';
import { firstValueFrom } from 'rxjs';

@Injectable()
export class AiService {
  private readonly baseUrl = process.env.AI_SERVICE_URL ?? 'http://127.0.0.1:8200';
  constructor(private readonly http: HttpService) {}

  async predictWeek(input: {
    restaurantId: string; productId: string; title: string;
    category?: string; targetWeek: string; avgDailySales?: number;
  }) {
    const { data } = await firstValueFrom(
      this.http.post(`${this.baseUrl}/integration/restomind/predict`, input),
    );
    return data; // فيه predictedOrders, featuresUsed, factors...
  }
}
```

```typescript
// src/predictions/predictions.service.ts  (المنطق: نده الموديل → خزّن)
async recalculate(restaurantId: string, product: Product, targetWeek: string) {
  const pred = await this.aiService.predictWeek({
    restaurantId,
    productId: String(product._id),
    title: product.title,
    category: await this.categoryName(product.category),
    targetWeek,
    avgDailySales: 60, // أو تقدير المالك المخزّن
  });

  return this.predictionModel.findOneAndUpdate(
    { restaurantId, productId: product._id, targetWeek },
    {
      restaurantId, productId: product._id,
      modelVersionId: pred.modelVersionId,
      targetWeek: pred.targetWeek,
      predictedOrders: pred.predictedOrders,
      featuresUsed: pred.featuresUsed,
      actualOrders: null,
    },
    { upsert: true, new: true },
  );
}
```

> **مرجع كامل شغّال:** ملف `app/integration/connect_restomind.py` عندنا بيعمل بالظبط
> السيناريو ده بالبايثون (يقرأ من Mongo → ينده الموديل → يكتب في `predictions`). استخدموه
> كمرجع للـ NestJS.

---

## 5. الفرونت اند — إزاي يعرض التوقّعات

الفرونت **ما بينداش الموديل**. بينده الباك:

```typescript
// شاشة الأدمن: خطة الإنتاج
const res = await fetch(`${API}/predictions?restaurantId=${rid}&week=2025-03-10`);
const predictions = await res.json();
// اعرض لكل منتج: predictedOrders + السبب (factors) + مستوى الثقة (confidence)
```

**اعرضوا دايمًا الـ `factors`** جنب الرقم (مثلاً «كنافة ٧٥٠ — رمضان +١٥٠٪») — ده اللي بيخلّي
المدير يثق في الرقم. ومستوى الثقة (`confidence`) لو `low` اعرضوه كتحذير «تقدير مبدئي».

---

## 6. الـ Seed — عشان المشروع يبقى متطابق عند الجميع ⭐

عشان كل واحد في الفريق يشوف نفس الأرقام في الديمو، **كلكم تعملوا seed بنفس الداتا**.
في مجلد الموديل فيه سكريبت جاهز بيملأ الـ MongoDB بمخبز كامل (١١ صنف + سنتين مبيعات):

### الخطوات (مرة واحدة، لأي حد في الفريق)

```bash
# 1) شغّلوا MongoDB (نفس الـ DB اللي الباك بيستخدمه)
mongod --dbpath <مجلد-داتا> --port 27017

# 2) في الباك: .env فيه
#    DB_URL=mongodb://127.0.0.1:27017/restomind

# 3) شغّلوا الموديل
cd model && REGISTRY_STORE=data/registry.json \
  .venv/bin/uvicorn app.api.main:app --port 8200 &

# 4) ⭐ seed المخبز الكامل في نفس الـ Mongo
MONGO_URL=mongodb://127.0.0.1:27017/restomind \
  .venv/bin/python -m app.integration.seed_bakery_history
#   بيدخّل: 11 منتج (كل واحد فيه field اسمه sku) + 8,019 sales_transactions

# 5) ⭐ شغّلوا الوصلة: تقرأ المنتجات → تنده الموديل → تكتب predictions
MONGO_URL=mongodb://127.0.0.1:27017/restomind MODEL_URL=http://127.0.0.1:8200 \
  .venv/bin/python -m app.integration.connect_restomind 2025-03-10
```

بعد كده الـ MongoDB بتاعكم كلكم فيه **نفس** الـ:
- `restaurants` (مخبز واحد)
- `products` (١١ صنف، كل واحد مربوط بـ `sku`)
- `sales_transactions` (سنتين مبيعات = التاريخ اللي الموديل اتعلّم منه)
- `predictions` (توقّعات جاهزة، بالتقويم المصري)

### ليه الـ `sku` field مهم؟
المنتجات اللي عليها `sku` (زي `SWEET_KONAFA`) بيوجّهها الكونيكتور للموديل **المدرَّب**
(`/forecast/weekly` — دقة عالية، بيعرف رمضان). المنتجات من غير `sku` بتروح للموديل
البدائي (rule-based). فالـ seed بيخلّي كل المنتجات مربوطة بالموديل المدرَّب.

### التحقق (كلكم لازم تشوفوا نفس الأرقام)
```bash
mongosh --port 27017 restomind --eval 'db.predictions.find({},{targetWeek:1,predictedOrders:1}).pretty()'
```
مثلاً كنافة في أسبوع رمضان ≈ ١١٠٠، كرواسون ≈ ١٠٠٠ (بيقل)، عيش بلدي ≈ ١٢٤٠٠.

---

## 7. الفرق بين الموديلين (مهم تفهموه)

| | rule-based (بدائي) | trained (مدرَّب) |
|---|---|---|
| متى؟ | منتج جديد بلا تاريخ | منتج عليه `sku` أو عنده 90 يوم مبيعات |
| Endpoint | `/integration/restomind/predict` | `/forecast/weekly` (بـ sku) |
| الدقة | تقديرية (قواعد + فئة) | عالية (WAPE ~16%) |
| `modelVersionId` | `restomind-bridge/rule_based-v0.1` | `calendar_decomposed/batch` |

الموديل بيتحوّل من البدائي للمدرَّب **تلقائيًا** لكل صنف بعد ما يجمّع 90 يوم مبيعات
(عبر `/data/ingest` أو `/integration/restomind/ingest`).

### إعادة التدريب (Retrain) — مسؤولية مين؟

- **الموديل ما بيسحبش من الباك.** الباك هو اللي **بيدفع** المبيعات لـ `/data/ingest`.
- إعادة التدريب بتحصل **تلقائيًا جوّه استدعاء `/data/ingest`** (مش موقّت داخل الموديل).
- **الجدولة (كل ليلة) مسؤوليتكم في الباك** عبر `@nestjs/schedule`:

```typescript
import { Cron, CronExpression } from '@nestjs/schedule';

@Cron(CronExpression.EVERY_DAY_AT_2AM)
async nightlySync() {
  const sales = await this.salesService.getYesterday();   // من sales_transactions
  await this.aiService.ingest(sales);                     // POST /data/ingest → retrain تلقائي
  // بعد كده أعيدوا حساب predictions للأسبوع الجاي (recalculate)
}
```
الرد من `/data/ingest` بيقول `model_retrained: true/false` و `newly_switched_to_ml: [...]`.

---

## 8. ملخص المسؤوليات

| الطرف | مسؤول عن |
|---|---|
| **الباك اند** | ينده الموديل، يخزّن في `predictions` و`offers`، يبعت المبيعات اليومية للموديل (`/ingest`) |
| **الفرونت اند** | يقرأ `predictions`/`offers` من الباك ويعرضها + الأسباب + مستوى الثقة |
| **خدمة الموديل** | تحسب التوقّع (تقويم مصري) وترجّعه — ما بتلمسش قاعدة البيانات بتاعتكم |
| **الجميع** | يعمل seed بنفس السكريبت (`seed_bakery_history.py`) عشان الأرقام تبقى متطابقة |

---

## 9. مهام تيم الباك اند (Checklist — زي GitHub Issues) ⭐

> انسخوا كل واحدة كـ Issue على جيتهاب. مرتّبين بالأولوية. الموديل جاهز — دي مهامكم إنتوا.

### 🎫 Issue #1 — إعداد الاتصال بخدمة الموديل
- [ ] ضيفوا `AI_SERVICE_URL=http://127.0.0.1:8200` في `.env`
- [ ] ثبّتوا `@nestjs/axios`: `npm i @nestjs/axios axios`
- [ ] أنشئوا `AiService` (الكود في §4) — دالة واحدة `predictWeek()` بتنده الموديل
- **تم لما:** تنده الموديل من الباك وترجع رد ناجح

### 🎫 Issue #2 — مجموعة predictions + endpoint الحساب
- [ ] أنشئوا `prediction.model.ts` بالحقول:
      `restaurantId, productId, modelVersionId, targetWeek, predictedOrders, featuresUsed, actualOrders?`
- [ ] `POST /predictions/recalculate` → ينده `AiService.predictWeek()` → يخزّن النتيجة (upsert)
- [ ] `GET /predictions?restaurantId=&week=` → للفرونت
- **تم لما:** تعمل recalculate لمنتج وتلاقي document في `predictions` فيه `predictedOrders`

### 🎫 Issue #3 — الـ Cron الليلي (ده اللي بيخص الـ retrain) ⭐
- [ ] ثبّتوا `@nestjs/schedule`: `npm i @nestjs/schedule`
- [ ] Cron كل ليلة (§7): يجمع مبيعات امبارح من `sales_transactions`
- [ ] يبعتها `POST {AI_SERVICE_URL}/data/ingest`
- [ ] بعدها يعيد حساب `predictions` للأسبوع الجاي (recalculate)
- **ملاحظة:** إعادة التدريب بتحصل **تلقائيًا جوّه `/data/ingest`** — إنتوا بس بتبعتوا المبيعات
- **تم لما:** الـ Cron يشتغل، والرد يرجّع `model_retrained` و `newly_switched_to_ml`

### 🎫 Issue #4 — كتابة sales_transactions عند اكتمال الأوردر
- [ ] لما `PATCH /orders/:id/status` تحط `status: Delivered`
- [ ] اكتبوا صف في `sales_transactions` لكل صنف في الأوردر
      (`productId, date, quantity, source: marketplace_order, promotionActive`)
- **تم لما:** أوردر Delivered يعمل صفوف مبيعات — دي مصدر بيانات الـ Cron في Issue #3

### 🎫 Issue #5 — شاشات الأدمن والستورز
- [ ] الأدمن: `GET /predictions` أو نداء `/integration/restomind/production-plan`
- [ ] الستورز (قرب القفل): نداء `/integration/restomind/surplus-offers` → اعرضوا العروض
- [ ] **اعرضوا `factors` و `confidence`** جنب كل رقم (سبب + مستوى ثقة)
- **تم لما:** الشاشتين بتعرضوا أرقام الموديل مع أسبابها

### 🎫 Issue #6 — قياس الدقة (Feedback loop)
- [ ] Cron أسبوعي: لكل `prediction` عدّى أسبوعها، املوا `actualOrders` من `sales_transactions`
- [ ] احسبوا الخطأ (predicted مقابل actual) → `GET /predictions/accuracy`
- **تم لما:** تقدروا تجاوبوا «التوقّع كان قريب قد إيه؟» بأرقام حقيقية

### ملاحظات مهمة للتيم
- ✅ الموديل **stateless غالبًا** — ما بيسحبش من Mongo وما بيجدولش نفسه. إنتوا العقل المنظّم.
- ✅ الـ **retrain تلقائي** جوّه `/data/ingest` — مش محتاجين تعملوه بإيديكم.
- ✅ الفرونت **ما بيندهش الموديل** — بينده الباك بس.
- ✅ للتجربة: استوردوا `postman_collection.json` وشوفوا كل الطلبات شغّالة.

---

## 10. مراجع

- خطوات الربط اللايف كاملة ومجرّبة: `LIVE_DEMO.md`
- ملخص المشروع لأي مطوّر جديد: `HANDOFF.md`
- كل الطلبات جاهزة للتجربة: `postman_collection.json` (استوردوه في Postman)
- واجهة تفاعلية لكل الـ endpoints: شغّلوا الموديل وافتحوا `/docs`
