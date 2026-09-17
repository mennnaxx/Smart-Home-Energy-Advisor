import streamlit as st
import plotly.graph_objects as go
from advisor_logic import load_artifacts, run_smart_home_advisor, build_usage_hour_labels

st.set_page_config(page_title="Smart Home Energy Advisor", page_icon="⚡", layout="wide")

st.markdown("""
<style>
    .stApp {
        background-color: #e9f2ff;
    }
</style>
""", unsafe_allow_html=True)

model, scaler_aux, meta, level_table = load_artifacts()
APPLIANCES = meta["appliance_types"]
HOUR_LABELS = build_usage_hour_labels(level_table, meta)  # {appliance: {"Low": "...", "Medium": "...", "High": "..."}}

if "devices" not in st.session_state:
    st.session_state.devices = [{"appliance": APPLIANCES[0], "level": "Medium", "power_kw": None}]

st.title("Smart Home Energy Advisor")
st.caption("Estimate your household's energy consumption, electricity bill, and get budget-aware recommendations.")

# ---------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------
st.header("Household Details")
col1, col2 = st.columns(2)
with col1:
    household_size = st.number_input(
        "Number of people in the household", min_value=1, max_value=20, value=3, step=1
    )
with col2:
    user_budget = st.number_input(
        "Monthly electricity budget (EGP) — optional", min_value=0.0, value=0.0, step=50.0
    )

st.header("Appliances")

for i, device in enumerate(st.session_state.devices):
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
        appliance = c1.selectbox(
            "Device", APPLIANCES, index=APPLIANCES.index(device["appliance"]), key=f"appliance_{i}"
        )

        # Usage level shown to the user as an approximate average number of
        # hours of daily use — never as kWh, which means nothing to a
        # regular user picking a level.
        level_labels = HOUR_LABELS[appliance]
        level_choice = c2.selectbox(
            "Usage level", list(level_labels.values()),
            index=["Low", "Medium", "High"].index(device["level"]), key=f"level_{i}"
        )
        level = [k for k, v in level_labels.items() if v == level_choice][0]

        power_kw = c3.number_input(
            "Power (kW) — optional", min_value=0.0, value=0.0, step=0.1, key=f"power_{i}"
        )

        st.session_state.devices[i] = {
            "appliance": appliance,
            "level": level,
            "power_kw": power_kw if power_kw > 0 else None,
        }

        if len(st.session_state.devices) > 1:
            if c4.button("✕ Remove", key=f"remove_{i}"):
                st.session_state.devices.pop(i)
                st.rerun()

if st.button("＋ Add another device"):
    st.session_state.devices.append({"appliance": APPLIANCES[0], "level": "Medium", "power_kw": None})
    st.rerun()

st.divider()

# ---------------------------------------------------------------
# Run prediction
# ---------------------------------------------------------------
if st.button("Calculate Energy Advisor Report", type="primary"):
    payload = run_smart_home_advisor(
        st.session_state.devices, household_size,
        user_budget if user_budget > 0 else None,
        model, scaler_aux, meta, level_table,
    )

    st.header(" Your Energy Advisor Dashboard")

    # ---- KPI row ----
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Predicted Consumption", f"{payload['predicted_kwh']} kWh")
    m2.metric("Estimated Bill", f"{payload['estimated_bill']} EGP")
    m3.metric("Model Used", payload["best_model_name"])
    m4.metric("Model Accuracy (R²)", f"{payload['accuracy_r2']:.3f}")

    st.markdown("")

    # ---- Charts row: per-device breakdown + budget gauge ----
    chart_col1, chart_col2 = st.columns([1.3, 1])

    with chart_col1:
        st.subheader("Consumption by device")
        devices_sorted = sorted(payload["device_results"], key=lambda d: d["predicted_kwh"], reverse=True)
        names = [d["appliance"] for d in devices_sorted]
        values = [d["predicted_kwh"] for d in devices_sorted]

        fig_donut = go.Figure(data=[go.Pie(
            labels=names, values=values, hole=0.55,
            textinfo="label+percent", sort=False,
        )])
        fig_donut.update_layout(
            margin=dict(t=10, b=10, l=10, r=10), height=340,
            showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.25),
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with chart_col2:
        st.subheader("Budget usage")
        if payload["budget_status"]:
            pct = (payload["estimated_bill"] / user_budget) * 100 if user_budget else 0.0
            gauge_color = {"Within Budget": "#2ecc71", "Close to Budget": "#f39c12", "Over Budget": "#e74c3c"}.get(
                payload["budget_status"], "#95a5a6"
            )
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pct,
                number={"suffix": "%"},
                gauge={
                    "axis": {"range": [0, max(150, pct + 10)]},
                    "bar": {"color": gauge_color},
                    "steps": [
                        {"range": [0, 90], "color": "#eafaf1"},
                        {"range": [90, 110], "color": "#fef5e7"},
                        {"range": [110, max(150, pct + 10)], "color": "#fdedec"},
                    ],
                    "threshold": {"line": {"color": "black", "width": 3}, "thickness": 0.8, "value": 100},
                },
            ))
            fig_gauge.update_layout(margin=dict(t=30, b=10, l=20, r=20), height=340)
            st.plotly_chart(fig_gauge, use_container_width=True)
        else:
            st.info("Enter a monthly budget above to see budget usage, recommendations, and a what-if scenario.")

    # ---- Budget status text ----
    if payload["budget_status"]:
        status_color = {
            "Within Budget": "green",
            "Close to Budget": "orange",
            "Over Budget": "red",
        }.get(payload["budget_status"], "gray")
        st.markdown(f"### Budget Status: :{status_color}[{payload['budget_status']}]")
        st.write(payload["budget_message"])
        if payload.get("cost_per_person") is not None:
            st.caption(f"Estimated cost per person: {payload['cost_per_person']} EGP")

    # ---- Per-device breakdown table ----
    with st.expander("Per-device breakdown", expanded=False):
        st.table([
            {
                "Device": d["appliance"],
                "Usage level": d["level"],
                "Hours/day": d["hours_per_day"] if d["hours_per_day"] is not None else "—",
                "Daily kWh": d["daily_kwh"],
                "Context factor": d["context_factor"],
                "Monthly kWh": d["predicted_kwh"],
            }
            for d in payload["device_results"]
        ])

    # ---- Recommendations ----
    if payload["recommendations"]:
        st.subheader(" Recommendations")
        for r in payload["recommendations"]:
            st.write(f"- {r}")

    if payload["what_if_scenario"]:
        st.subheader(" What-if Scenario")
        st.write(payload["what_if_scenario"])