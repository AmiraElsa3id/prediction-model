"""Investor demo dashboard.

Run:  .venv/bin/streamlit run dashboard.py

Walks the five demo beats from the plan:
  1. "It knows Egypt"        -- forecast chart with Ramadan/Eid visibly baked in
  2. "It predicts tomorrow"  -- per-item quantity with a confidence interval
  3. "It stops waste"        -- manual entry vs forecast, projected EGP loss
  4. "It sells the surplus"  -- near-closing detection -> Arabic offer
  5. "The money slide"       -- simulated waste avoided and EGP saved

Everything on screen is labelled SIMULATED. The number an investor remembers is a
projection from generated data, and saying so plainly is both honest and more credible
than a bare figure nobody can trace to a real bakery.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.core.egypt_calendar import CALENDAR
from app.core.evaluation import make_folds
from app.core.features import build_features
from app.core.generate import generate
from app.core.items import BY_SKU, CATALOGUE
from app.core.surplus import detect_surplus
from app.models.forecaster import CalendarDecomposed, MovingAverage, SeasonalNaive
from app.models.service import ForecastService

st.set_page_config(page_title="Bakery AI — Demo", page_icon="🥐", layout="wide")

PRIMARY = "#C8853C"
DANGER = "#C0392B"
OK = "#27AE60"


# --------------------------------------------------------------------------------------
# Cached heavy work
# --------------------------------------------------------------------------------------


@st.cache_resource(show_spinner="Training the forecaster on simulated data…")
def load_service() -> ForecastService:
    return ForecastService(horizon=1).train()


@st.cache_data(show_spinner="Generating simulated bakery history…")
def load_raw() -> pd.DataFrame:
    return generate()


def sku_label(sku: str) -> str:
    item = BY_SKU[sku]
    return f"{item.name_ar} · {item.name_en}"


# --------------------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------------------

st.title("🥐 Bakery Demand & Surplus AI")
st.caption("Demand forecasting and automated surplus detection for Egyptian bakeries")

st.warning(
    "**All figures below are SIMULATED.** The bakery is pre-launch, so the system is "
    "trained on a synthetic dataset generated with known Egyptian-calendar effects. "
    "These are projections of what it would do once connected to a real POS — not "
    "measurements from a real bakery.",
    icon="⚠️",
)

service = load_service()
raw = load_raw()

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 It knows Egypt", "🚨 It stops waste", "🏷️ It sells the surplus", "💰 The money",
])


# --------------------------------------------------------------------------------------
# Beat 1 + 2 — forecast that visibly knows the Egyptian calendar
# --------------------------------------------------------------------------------------

with tab1:
    st.subheader("The forecast anticipates Ramadan, Eid and the weekend")
    col_ctl, col_metric = st.columns([2, 3])

    with col_ctl:
        sku = st.selectbox(
            "Item", [i.sku for i in CATALOGUE], format_func=sku_label, index=2,
        )
        window = st.selectbox(
            "Period",
            ["Ramadan → Eid 2025 (Feb–Apr)", "Full year 2024", "School vs summer 2024"],
        )

    ranges = {
        "Ramadan → Eid 2025 (Feb–Apr)": ("2025-02-15", "2025-04-20"),
        "Full year 2024": ("2024-01-01", "2024-12-31"),
        "School vs summer 2024": ("2024-05-15", "2024-10-15"),
    }
    start, end = ranges[window]
    dates = pd.date_range(start, end, freq="D")

    # Forecast across the window. Actual sales come from the generated history for context.
    preds = [service.forecast(sku, d.date()) for d in dates]
    fc = pd.DataFrame({
        "date": dates,
        "forecast": [p.quantity for p in preds],
        "lower": [p.lower for p in preds],
        "upper": [p.upper for p in preds],
    })
    actual = (
        raw[(raw["sku"] == sku)][["date", "sales_qty", "true_demand"]]
        .assign(date=lambda d: pd.to_datetime(d["date"]))
    )
    fc = fc.merge(actual, on="date", how="left")

    cal = CALENDAR.feature_frame(start, end)
    cal["date"] = pd.to_datetime(cal["date"])

    fig = go.Figure()
    # Shade Ramadan and mark Eid so the calendar is legible at a glance.
    ram = cal[cal["is_ramadan"] == 1]["date"]
    if len(ram):
        fig.add_vrect(x0=ram.min(), x1=ram.max(), fillcolor="#8E44AD", opacity=0.10,
                      line_width=0, annotation_text="Ramadan", annotation_position="top left")
    for eid in cal[cal["eid_fitr_day_index"] == 1]["date"]:
        fig.add_vline(x=eid, line_dash="dot", line_color="#8E44AD",
                      annotation_text="Eid", annotation_position="top")

    fig.add_trace(go.Scatter(
        x=fc["date"], y=fc["upper"], mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=fc["date"], y=fc["lower"], mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(200,133,60,0.18)",
        name="Confidence range"))
    fig.add_trace(go.Scatter(
        x=fc["date"], y=fc["true_demand"], mode="lines",
        line=dict(color="#95A5A6", width=1, dash="dot"), name="Actual demand (sim)"))
    fig.add_trace(go.Scatter(
        x=fc["date"], y=fc["forecast"], mode="lines",
        line=dict(color=PRIMARY, width=2.5), name="AI forecast"))
    fig.update_layout(
        height=420, margin=dict(t=30, b=10), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        yaxis_title="Units / day",
    )
    col_metric.plotly_chart(fig, use_container_width=True)

    # Beat 2: pick one day and show the explained recommendation.
    st.divider()
    st.subheader("Tomorrow's recommendation, explained")
    c1, c2 = st.columns([1, 2])
    pick = c1.date_input("Date", value=dt.date(2025, 3, 15),
                         min_value=dt.date(2024, 1, 1), max_value=dt.date(2025, 6, 30))
    r = service.forecast(sku, pick)
    c1.metric(f"Produce — {BY_SKU[sku].name_ar}", f"{r.quantity} units",
              help="Point forecast at the item's profit-optimal service level")
    c1.caption(f"Range {r.lower}–{r.upper} · confidence: {r.confidence}")

    if r.factors:
        fdf = pd.DataFrame(r.factors)
        bar = go.Figure(go.Bar(
            x=fdf["impact_pct"], y=fdf["factor"], orientation="h",
            marker_color=[OK if v > 0 else DANGER for v in fdf["impact_pct"]],
        ))
        bar.update_layout(height=260, margin=dict(t=10, b=10),
                          xaxis_title="Impact on demand (%)", title="Why this number")
        c2.plotly_chart(bar, use_container_width=True)
    else:
        c2.info("An ordinary day — no special calendar drivers active.")


# --------------------------------------------------------------------------------------
# Beat 3 — waste-prevention alert
# --------------------------------------------------------------------------------------

with tab2:
    st.subheader("Catch over-production before it becomes waste")
    st.write("The manager enters a production quantity. The AI flags it when it exceeds "
             "what demand can plausibly absorb, and prices the likely loss.")

    c1, c2 = st.columns([1, 2])
    wsku = c1.selectbox("Item", [i.sku for i in CATALOGUE], format_func=sku_label,
                        index=4, key="waste_sku")
    wdate = c1.date_input("Date", value=dt.date(2025, 2, 18), key="waste_date")
    fc_w = service.forecast(wsku, wdate)
    default_plan = int(fc_w.upper * 1.8)
    plan = c1.number_input("Manager's planned quantity", min_value=0,
                           value=default_plan, step=5)

    alert = service.waste_alert(wsku, wdate, plan)
    colour = {"none": OK, "low": "#F39C12", "medium": "#E67E22", "high": DANGER}[alert["severity"]]

    c2.markdown(f"### <span style='color:{colour}'>{alert['severity'].upper()}</span>",
                unsafe_allow_html=True)
    c2.write(alert["message"])
    m1, m2, m3 = c2.columns(3)
    m1.metric("AI forecast", f"{alert['forecast_qty']}")
    m2.metric("Plausible max", f"{alert['forecast_upper']}")
    m3.metric("Projected waste", f"{alert['projected_waste_cost_egp']:,.0f} EGP",
              delta=f"{alert['excess_qty']} units", delta_color="inverse")
    if alert["factors"]:
        c2.caption("Drivers: " + ", ".join(
            f"{f['factor']} {f['impact_pct']:+.0f}%" for f in alert["factors"]))


# --------------------------------------------------------------------------------------
# Beat 4 — surplus -> Arabic offer
# --------------------------------------------------------------------------------------

with tab3:
    st.subheader("Turn end-of-day surplus into an automatic offer")

    c1, c2 = st.columns([1, 2])
    hour = c1.slider("Time of day", 12, 21, 19)
    now = dt.datetime.combine(dt.date(2025, 2, 18), dt.time(hour, 0))

    st.session_state.setdefault("stock", {
        "CAKE_GATEAU": 45, "PASTRY_CROISSANT": 30, "SWEET_KONAFA": 20,
        "SAVOURY_FETEER": 15, "BREAD_BALADI": 200,
    })
    c1.caption("Unsold stock on the shelf right now:")
    stock = {}
    for s, default in st.session_state["stock"].items():
        stock[s] = c1.number_input(BY_SKU[s].name_ar, 0, 999, default, key=f"st_{s}")

    daily = {s: service.forecast(s, now.date()).quantity for s in stock}
    surplus = detect_surplus(stock, daily, now)

    if not surplus:
        c2.success("Nothing at risk yet — plenty of the day left to sell through.")
    else:
        c2.write(f"**{len(surplus)} item(s) at risk** · "
                 f"total value at risk: **{sum(s.value_at_risk_egp for s in surplus):,.0f} EGP**")
        for s in surplus:
            with c2.container(border=True):
                cc1, cc2 = st.columns([1, 2])
                cc1.metric(BY_SKU[s.sku].name_ar, f"{s.current_stock} left",
                           delta=f"{s.suggested_discount_pct}% off", delta_color="off")
                cc1.caption(f"Urgency: {s.urgency} · {s.value_at_risk_egp:,.0f} EGP at risk")
                cc2.markdown(f"##### {BY_SKU[s.sku].name_ar}")
                cc2.caption(
                    f"{s.expected_remaining_sales:.0f} expected sales left tonight · "
                    f"suggested discount {s.suggested_discount_pct}%"
                )


# --------------------------------------------------------------------------------------
# Beat 5 — the money slide
# --------------------------------------------------------------------------------------


@st.cache_data(show_spinner="Running the business simulation…")
def run_simulation() -> dict:
    feats = build_features(raw, horizon=1)
    folds = make_folds(feats["date"], n_folds=6, test_days=28, horizon=1)
    models = {
        "Manager (today)": None,
        "Seasonal naive": SeasonalNaive(),
        "AI (decomposed)": CalendarDecomposed(),
    }
    test_frames, preds = [], {k: [] for k in models if models[k] is not None}
    for fold in folds:
        tr = feats[feats["date"] <= fold.train_end]
        te = feats[(feats["date"] >= fold.test_start) & (feats["date"] <= fold.test_end)]
        te = te[te["true_demand"].notna() & (te["is_closed"] == 0)]
        if tr.empty or te.empty:
            continue
        test_frames.append(te)
        for name, m in models.items():
            if m is not None:
                preds[name].append(np.asarray(m.fit_predict(tr, te), dtype=float))
    ev = pd.concat(test_frames, ignore_index=True)
    demand = ev["true_demand"].to_numpy(float)
    skus = ev["sku"]

    def cost(production):
        c = np.zeros(len(production))
        waste = np.zeros(len(production))
        for s in skus.unique():
            it = BY_SKU[s]
            sel = (skus == s).to_numpy()
            over = np.maximum(production[sel] - demand[sel], 0)
            under = np.maximum(demand[sel] - production[sel], 0)
            waste[sel] = over
            c[sel] = over * it.unit_cost * it.spoilage_severity + under * it.shortage_cost
        return c.sum(), waste.sum()

    days = ev["date"].nunique()
    out = {}
    mgr_cost, mgr_waste = cost(ev["production_qty"].to_numpy(float))
    out["Manager (today)"] = (mgr_cost, mgr_waste)
    for name in preds:
        out[name] = cost(np.concatenate(preds[name]))
    return {"results": out, "days": days,
            "waste_prod": float(ev["production_qty"].sum())}


with tab4:
    st.subheader("What it saves — simulated over the backtest window")
    sim = run_simulation()
    res, days = sim["results"], sim["days"]
    mgr_cost, mgr_waste = res["Manager (today)"]
    ai_cost, ai_waste = res["AI (decomposed)"]

    saving_month = (mgr_cost - ai_cost) / days * 30
    waste_cut = (mgr_waste - ai_waste) / mgr_waste * 100

    m1, m2, m3 = st.columns(3)
    m1.metric("Monthly saving (1 branch)", f"{saving_month:,.0f} EGP",
              help="Waste + lost-sales cost avoided vs. the manager's actual production")
    m2.metric("Waste reduction", f"{waste_cut:.0f}%", delta="fewer units binned",
              delta_color="off")
    m3.metric("Annual saving (1 branch)", f"{saving_month * 12:,.0f} EGP")

    names = list(res)
    costs = [res[n][0] / days for n in names]
    fig = go.Figure(go.Bar(
        x=names, y=costs,
        marker_color=[DANGER, "#95A5A6", OK],
        text=[f"{c:,.0f}" for c in costs], textposition="outside",
    ))
    fig.update_layout(height=380, yaxis_title="Cost per day (EGP)",
                      title="Daily cost of waste + lost sales, by production policy",
                      margin=dict(t=50))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Simulated over {days} days × {len(BY_SKU)} items, rolling-origin backtest. "
               "Costs include wasted production and lost sales (with goodwill). "
               "Figures scale with branch count and depend on real unit economics.")
