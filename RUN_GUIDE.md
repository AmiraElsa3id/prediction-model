# دليل تشغيل المشروع + شرح الهيكل (Model / AI Service)

> خدمة الذكاء الاصطناعي للتنبؤ بإنتاج المخابز المصرية وتسويق الفائض.
> مبنية بـ **Python + FastAPI**، بتشتغل كـ microservice منفصل بيتكلم مع نظام RestoMind.
>
> ⚠️ **مهم:** كل الأرقام دلوقتي **محاكاة (SIMULATED)** — الموديل متدرّب على داتا مولّدة، مش
> داتا مخبز حقيقي. أي رقم يتعرض للجنة/العميل لازم يتقال إنه «إسقاط» مش «قياس».

---

## 1. المتطلبات (Prerequisites)

| الأداة | الإصدار | ليه |
|---|---|---|
| Python | 3.11+ (اتجرّب على 3.13) | كل كود الموديل |
| pip / venv | مع بايثون | تثبيت المكتبات |
| (اختياري) Node.js 18+ | | لتشغيل اختبارات Postman عبر `npx newman` |
| (اختياري) MongoDB + mongosh | 6+ | للربط اللايف مع باك RestoMind |

لتأكيد إن بايثون متثبّت:
```bash
python3 --version      # لازم يطلع 3.11 أو أحدث
```

---

## 2. التثبيت (مرة واحدة)

من داخل مجلد المشروع (`model/`):

```bash
# 1) اعمل بيئة افتراضية معزولة
python3 -m venv .venv

# 2) ثبّت كل المكتبات المطلوبة
.venv/bin/pip install -r requirements.txt
```

> كل الأوامر بعد كده بتستخدم `.venv/bin/python` و `.venv/bin/uvicorn` — عشان تضمني إنك
> شغّالة داخل البيئة المعزولة، من غير ما تعملي activate.

---

## 3. تشغيل المشروع — 4 طرق حسب اللي محتاجاه

### أ) تشغيل السيرفر (الأهم) — الـ API

```bash
.venv/bin/uvicorn app.api.main:app
```
- بيستنى ~20 ثانية (الموديل بيتدرّب عند الإقلاع)، وبعدين بتظهر رسالة
  `Application startup complete`.
- افتحي **http://127.0.0.1:8000/docs** → دي واجهة Swagger، تقدري تجرّبي أي endpoint
  بـ **Try it out** → **Execute**.
- لتغيير البورت: `--port 8200`.
- متغيّرات بيئة اختيارية:
  - `COLD_START=true` → يبدأ بدون أي بيانات (كله rule-based) — لعرض سيناريو مخبز جديد.
  - `REGISTRY_STORE=data/registry.json` → يحفظ ما تعلّمه لكل مطعم بعد الريستارت.

### ب) توليد الداتا المحاكاة من جديد

```bash
.venv/bin/python -m app.core.generate
# بيكتب data/synthetic_pos.parquet (سنتين، 11 صنف، ~8000 صف)
```

### ج) تشغيل الاختبارات (للتأكد إن كل حاجة سليمة)

```bash
.venv/bin/python -m pytest tests/ -q
# المفروض يطلع: 83 passed (بياخد ~3-4 دقايق لأنه بيدرّب موديلات حقيقية)
```

### د) عرض الأرقام (backtest + محاكاة الأعمال)

```bash
.venv/bin/python -m scripts.run_backtest     # جدول مقارنة الموديلات (WAPE)
.venv/bin/python -m scripts.run_simulation   # التوفير بالجنيه مقابل تخمين المدير
```

### (اختياري) الـ Dashboard البصري

```bash
.venv/bin/streamlit run dashboard.py         # يفتح على http://localhost:8501
```

---

## 4. هيكل المشروع (Structure)

```
model/
├── app/                          ← كل كود الخدمة
│   ├── core/                     ← المنطق الأساسي (بدون FastAPI)
│   │   ├── egypt_calendar.py     ← التقويم المصري: رمضان/عيد (هجري) + قبطي + مدارس + مرتبات + ويكند جمعة-سبت
│   │   ├── items.py              ← كتالوج الـ 11 صنف: أسعار، صلاحية، اقتصاديات (newsvendor q*)
│   │   ├── generate.py           ← مولّد الداتا المحاكاة (سنتين مبيعات بتأثيرات معروفة)
│   │   ├── features.py           ← تنظيف + هندسة features (lags/rolling/calendar) + معالجة نفاد المخزون
│   │   ├── evaluation.py         ← مقاييس WAPE/MASE/pinball + backtest
│   │   ├── surplus.py            ← كشف الفائض قرب القفل + مستويات الخصم
│   │   └── market_priors.py      ← حساسية المناسبات لكل فئة (يولّدها LLM أو الملف)
│   ├── models/                   ← الموديلات
│   │   ├── seasonality.py        ← تأثيرات التقويم (Ridge) — قلب الميزة المصرية
│   │   ├── forecaster.py         ← الموديل الإنتاجي CalendarDecomposed + الأساس seasonal-naive
│   │   ├── rule_based.py         ← موديل البداية (cold-start) بدون داتا
│   │   └── service.py            ← التوجيه الهجين (قواعد → تدريب بعد 90 يوم) + ingestion
│   ├── marketing/                ← تسويق الفائض
│   │   ├── copy.py               ← كتابة الإعلان بالعامية (LLM + قوالب احتياطية)
│   │   └── publisher.py          ← النشر على فيسبوك/انستجرام (معاينة افتراضيًا)
│   ├── integration/              ← ⭐ الربط مع RestoMind
│   │   ├── restomind.py          ← الجسر: بيحوّل شكل داتاهم لمخرجات الموديل
│   │   ├── registry.py           ← حالة كل مطعم لوحده (multi-tenant) + تعلّم المستوى
│   │   ├── seed_restomind.py     ← seed مطعم صغير للتجربة
│   │   ├── seed_bakery_history.py← ⭐ seed المخبز الكامل (11 صنف + سنتين مبيعات)
│   │   └── connect_restomind.py  ← ⭐ الوصلة الحية: Mongo → الموديل → predictions
│   └── api/
│       ├── schemas.py            ← نماذج الإدخال/الإخراج (Pydantic)
│       └── main.py               ← تطبيق FastAPI (16 endpoint) + CORS
├── scripts/
│   ├── run_backtest.py           ← مقارنة الموديلات
│   └── run_simulation.py         ← محاكاة التوفير بالجنيه
├── tests/                        ← 83 اختبار
├── data/
│   ├── synthetic_pos.parquet     ← الداتا المولّدة
│   ├── market_priors.json        ← حساسية المناسبات (نتيجة تحليل الـ LLM)
│   └── models/                   ← الموديلات المحفوظة
├── dashboard.py                  ← عرض Streamlit
├── postman_collection.json       ← 17 طلب جاهز (تستوردي في Postman)
├── requirements.txt              ← المكتبات
├── README.md                     ← نظرة عامة (إنجليزي)
├── RUN_GUIDE.md                  ← الملف ده
├── INTEGRATION_GUIDE.md          ← دليل ربط الفرونت والباك (بالتفصيل)
├── LIVE_DEMO.md                  ← خطوات الربط اللايف مع RestoMind
├── HANDOFF.md                    ← ملخص كامل لأي مطوّر/AI يكمّل
└── problem_analysis.html         ← صفحة عرض حجم المشكلة (للجنة/العميل)
```

---

## 5. أهم 3 ملفات تفهميها الأول

1. **`app/api/main.py`** — كل الـ endpoints (نقطة الدخول).
2. **`app/models/service.py`** — الموديل والتوجيه الهجين (قواعد ↔ تدريب).
3. **`app/integration/`** — كل حاجة خاصة بالربط مع RestoMind.

---

## 6. الـ Endpoints المتاحة (ملخص)

| المجموعة | الـ Endpoint | بيعمل إيه |
|---|---|---|
| ops | `GET /health` | حالة السيرفر |
| lifecycle | `GET /model/status` | كل صنف قواعد ولا موديل مدرَّب |
| lifecycle | `POST /data/ingest` | إدخال مبيعات (الموديل بيتعلّم) |
| forecasting | `POST /forecast/daily` · `/weekly` | توقّع صنف واحد |
| forecasting | `POST /forecast/daily-batch` · `/weekly-batch` | توقّع كل الأصناف مرة واحدة |
| forecasting | `POST /forecast/seasonality-adjustment` | أثر المناسبات على يوم |
| alerts | `POST /alerts/waste-prevention` | تحذير هدر |
| surplus | `POST /surplus/detect` | كشف الفائض |
| marketing | `POST /marketing/generate-offer` · `/publish` | إعلان عربي + نشر |
| **restomind** | `POST /integration/restomind/production-plan` | خطة إنتاج (شاشة الأدمن) |
| **restomind** | `POST /integration/restomind/surplus-offers` | فائض + عروض (شاشة الستورز) |
| **restomind** | `POST /integration/restomind/predict` | توقّع أسبوعي (لمجموعة predictions) |
| **restomind** | `POST /integration/restomind/ingest` | مبيعات مطعم (يتعلّم المستوى) |
| **restomind** | `GET /integration/restomind/status/{id}` | حالة منتجات المطعم |

تفاصيل كل واحد + أمثلة كاملة في `INTEGRATION_GUIDE.md`.

---

## 7. تجربة سريعة (Smoke test)

```bash
# 1) شغّلي السيرفر في تيرمينال
.venv/bin/uvicorn app.api.main:app

# 2) في تيرمينال تاني — اسألي عن كنافة في رمضان مقابل يوم عادي
curl -s -X POST http://127.0.0.1:8000/forecast/daily \
  -H "Content-Type: application/json" \
  -d '{"sku":"SWEET_KONAFA","date":"2025-03-15"}'      # رمضان — رقم عالي

curl -s -X POST http://127.0.0.1:8000/forecast/daily \
  -H "Content-Type: application/json" \
  -d '{"sku":"SWEET_KONAFA","date":"2025-02-11"}'      # عادي — رقم أقل
```
لو الكنافة في رمضان أعلى من اليوم العادي → كل حاجة شغّالة صح.

---

## 8. إعادة التدريب (Retrain) — إزاي بتتم بالظبط ⭐

### الآلية الحالية: **مدفوعة بالحدث (event-driven)، مش cron داخل الموديل**

- **مفيش scheduler/cron جوّه خدمة الموديل نفسها.** الموديل خدمة سلبية (passive) —
  بيستنى الباك يبعتله البيانات، **مش بيسحب من الباك**.
- إعادة التدريب بتحصل **جوّه استدعاء `POST /data/ingest`**. أول ما الباك يبعت مبيعات
  جديدة، الموديل بيعمل الآتي **في نفس الطلب**:
  1. يضيف المبيعات للتاريخ المتجمّع
  2. يعيد حساب عدد أيام كل صنف
  3. يحدّث مستويات القواعد فورًا
  4. **يعيد تدريب الموديل الكامل** (`_fit_ml`) لو: فيه صنف واحد على الأقل عدّى الـ 90 يوم،
     و(صنف جديد لسه عدّى الحد **أو** فيه موديل متدرّب موجود يتحدّث)
  5. يرجّع في الرد `model_retrained: true/false` و `newly_switched_to_ml: [...]`

الكود المسؤول: `app/models/service.py` → دالة `ingest()`.

### يعني الـ "cron" فين؟ → **في الباك اند، مش في الموديل**

النمط الصح للإنتاج:
```
كل ليلة → الباك اند (NestJS @nestjs/schedule) → POST /data/ingest بمبيعات اليوم
        → الموديل يعيد التدريب تلقائيًا كجزء من الاستدعاء ده
```
يعني الجدولة (كل ليلة مثلاً) مسؤولية **الباك اند** — عنده `@nestjs/schedule` جاهز.
الموديل بيعيد التدريب كـ **side-effect** لوصول البيانات، مش بموقّت خاص بيه.

### مثال: الباك يجدول إرسال المبيعات كل ليلة

```typescript
// في الباك اند (NestJS)
import { Cron, CronExpression } from '@nestjs/schedule';

@Cron(CronExpression.EVERY_DAY_AT_2AM)          // كل يوم 2 صباحًا
async nightlySync() {
  const sales = await this.getYesterdaySales();  // من sales_transactions
  await this.aiService.ingest(sales);            // POST /data/ingest → retrain تلقائي
}
```

### فيه مسارين للـ ingest (مهم تفرّقي بينهم)

| المسار | Endpoint | بيعمل إيه |
|---|---|---|
| الأصناف المعروفة (الـ 11 SKU) | `POST /data/ingest` | **إعادة تدريب كاملة** للموديل CalendarDecomposed |
| منتجات أي مطعم (بالـ productId) | `POST /integration/restomind/ingest` | **يتعلّم المستوى** لكل منتج (مش تدريب كامل — لسه قواعد) |

### ملخص الإجابة

- ✅ **آه، الـ retrain متعامل معاه** — بيحصل تلقائيًا عند إدخال بيانات جديدة.
- ❌ **مش cron داخل الموديل** — الموديل ما بيسحبش من الباك، الباك هو اللي **بيدفع** البيانات.
- 🕐 **الجدولة (كل فترة) مسؤولية الباك اند** (`@nestjs/schedule`) اللي بينده `/data/ingest`.

> لو عايزين الموديل **يجدول نفسه** (in-app scheduler بدل الاعتماد على الباك)، ده ممكن
> يتضاف بمكتبة `APScheduler` — بس النمط الأنضف هو الباك يدفع، لأنه هو اللي عنده البيانات.

---

## 9. إيقاف كل حاجة

```bash
pkill -f "uvicorn app.api.main"     # السيرفر
pkill -f "streamlit run"            # الـ dashboard
```

---

## 10. مشاكل شائعة

| المشكلة | الحل |
|---|---|
| `ModuleNotFoundError` | نسيتي `.venv/bin/` قبل الأمر، أو التثبيت ما تمّش |
| `/health` بيرجّع "training" | استني ~20 ثانية، الموديل لسه بيتدرّب |
| البورت مشغول | استخدمي `--port 8200` مثلاً |
| الاختبارات بطيئة | طبيعي (~3-4 دقايق) لأنها بتدرّب موديلات حقيقية |
