"""
Core prediction / business logic for the Smart Home Energy Advisor.
Loaded once by the Streamlit app; no retraining happens here.
"""
import json
from datetime import date

import joblib
import pandas as pd

ARTIFACTS_DIR = "artifacts"


def load_artifacts():
    model = joblib.load(f"{ARTIFACTS_DIR}/best_model.joblib")
    scaler_aux = joblib.load(f"{ARTIFACTS_DIR}/scaler_aux.joblib")
    with open(f"{ARTIFACTS_DIR}/meta.json") as f:
        meta = json.load(f)
    level_table = pd.read_csv(f"{ARTIFACTS_DIR}/appliance_levels.csv", index_col="Appliance Type")
    return model, scaler_aux, meta, level_table


DEFAULT_ALWAYS_ON_APPLIANCES = {"Fridge"}


# ---------------------------------------------------------------
# Low / Medium / High shown to the user as *hours of use*, not kWh.
#
# Internally we still predict and reason in kWh (that's what the model
# was trained on). But kWh means nothing to a regular user picking a
# usage level, so behind the scenes we show each appliance's Low/Medium/
# High as an "average hours of use" figure. Those hours/day numbers are
# NOT recomputed here -- they were already derived once, at training
# time, from real data (kWh_for_level / real power_kW) and saved into
# appliance_levels.csv (see train_model.py). Reading them from the same
# table used for the actual kWh math means the number shown to the user
# is always the exact number the calculation uses -- no separate,
# disconnected UI-only estimate.
#
# Appliances that run continuously with a duty cycle (e.g. a fridge
# compressor) are never shown as "X hours/day" -- that framing is
# physically misleading for them -- they get a plain consumption-level
# label instead (see meta['always_on_appliances']).
# ---------------------------------------------------------------
def _fmt_hours(h: float) -> str:
    """Turn a raw hour value into a friendly label (minutes if under 1h)."""
    if h < 1:
        minutes = max(5, round(h * 60 / 5) * 5)
        return f"{minutes} min"
    rounded = round(h * 2) / 2  # nearest half hour
    if rounded == int(rounded):
        return f"{int(rounded)} hr" if rounded == 1 else f"{int(rounded)} hrs"
    return f"{rounded:g} hrs"


def build_usage_hour_labels(level_table, meta):
    """
    Returns: {appliance: {"Low": "...", "Medium": "...", "High": "..."}}
    Hours-based labels for normal appliances; plain consumption-level
    labels (no hours claim) for always-on/duty-cycle appliances.
    """
    always_on = set(meta.get("always_on_appliances", DEFAULT_ALWAYS_ON_APPLIANCES))
    labels = {}

    for appliance in level_table.index:
        if appliance in always_on:
            labels[appliance] = {
                "Low": "Low (efficient / well-maintained unit)",
                "Medium": "Medium (typical usage)",
                "High": "High (older unit / frequent door opening)",
            }
            continue

        low_h = level_table.loc[appliance, "low_hours"]
        med_h = level_table.loc[appliance, "medium_hours"]
        high_h = level_table.loc[appliance, "high_hours"]

        boundary_low_med = (low_h + med_h) / 2
        boundary_med_high = (med_h + high_h) / 2

        labels[appliance] = {
            "Low": f"Low (~under {_fmt_hours(boundary_low_med)}/day)",
            "Medium": f"Medium (~{_fmt_hours(boundary_low_med)}\u2013{_fmt_hours(boundary_med_high)}/day)",
            "High": f"High (~over {_fmt_hours(boundary_med_high)}/day)",
        }

    return labels


def _current_season(month: int) -> str:
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Fall"


def build_context():
    """Default context features the GUI does not ask the user for
    (outdoor temperature, season, day of week, weekend flag), derived
    from today's date and the dataset's own seasonal averages."""
    today = date.today()
    season = _current_season(today.month)
    return {
        "month": today.month,
        "season": season,
        "day_of_week": today.strftime("%A"),
        "is_weekend": int(today.weekday() >= 5),
    }


def build_feature_row(appliance, household_size, meta, context, power_kw_override=None):
    cols = meta["feature_columns"]
    row = {c: 0 for c in cols}

    row["Outdoor Temperature (°C)"] = meta["season_temp_avg"].get(context["season"], 15.0)
    row["Household Size"] = household_size
    row["Hour"] = meta["appliance_peak_hour"].get(appliance, 12)
    row["Month"] = context["month"]
    row["IsWeekend"] = context["is_weekend"]

    appl_col = f"Appliance Type_{appliance}"
    if appl_col in row:
        row[appl_col] = 1  # else it's the dropped baseline category (Air Conditioning)

    season_col = f"Season_{context['season']}"
    if season_col in row:
        row[season_col] = 1  # else baseline season (Fall)

    day_col = f"DayOfWeek_{context['day_of_week']}"
    if day_col in row:
        row[day_col] = 1  # else baseline day (Friday)

    aux = meta["appliance_aux_raw"].get(appliance, {
        "power_max": meta["power_max_median"],
        "available_duration": meta["available_duration_median"],
        "Auxiliary_Data_Available": 0,
    })
    power_max = power_kw_override * 1000 if power_kw_override else aux["power_max"]
    row["power_max"] = power_max
    row["available_duration"] = aux["available_duration"]
    row["Auxiliary_Data_Available"] = aux["Auxiliary_Data_Available"]

    return pd.DataFrame([row])[cols]


MIN_CONTEXT_FACTOR = 0.5
MAX_CONTEXT_FACTOR = 1.8


def predict_device_kwh(appliance, level, household_size, model, scaler_aux, meta, level_table, power_kw_override=None):
    """
    Predict monthly kWh for one device, built the physical way:

        monthly_kWh = power_kW * hours_per_day(appliance, level) * days_per_month

    Where:
      - hours_per_day comes straight from appliance_levels.csv (the same
        number shown to the user in the Low/Medium/High picker), derived
        from that appliance's real dataset1 consumption quantiles -- not
        a fixed 2/5/8-hour table and not an arbitrary ratio.
      - the ML model is used only to compute a bounded "context factor":
        how much higher/lower this specific context (household size,
        season, outdoor temperature, hour, month) pushes consumption
        relative to that appliance's own historical daily average. This
        is where the model's learned effects (appliance type, household
        size, season/temperature, time of day) actually influence the
        result, instead of the model output being blindly multiplied by
        a fixed 720-hour projection.

    For always-on / duty-cycle appliances (meta['always_on_appliances'],
    e.g. Fridge) there is no meaningful "hours/day" -- the level's real
    daily kWh from the data is used directly instead.
    """
    context = build_context()
    feat = build_feature_row(appliance, household_size, meta, context, power_kw_override)

    aux_cols = ["power_max", "available_duration"]
    feat_scaled = feat.copy()
    feat_scaled[aux_cols] = scaler_aux.transform(feat[aux_cols])

    context_pred = max(float(model.predict(feat_scaled)[0]), 0.0)

    known_appliance = appliance in level_table.index
    overall_mean = float(level_table.loc[appliance, "overall_mean_kwh"]) if known_appliance else (context_pred or 1.0)

    # How much does today's specific context (household size, season,
    # temperature, hour) push this appliance above/below its own
    # historical daily average? Bounded so an unusual context can't
    # swamp the Low/Medium/High choice the user actually made.
    context_factor = context_pred / overall_mean if overall_mean > 0 else 1.0
    context_factor = min(max(context_factor, MIN_CONTEXT_FACTOR), MAX_CONTEXT_FACTOR)

    days_per_month = meta.get("days_per_month", 30)
    always_on = set(meta.get("always_on_appliances", DEFAULT_ALWAYS_ON_APPLIANCES))

    if not known_appliance or appliance in always_on:
        level_kwh_base = float(level_table.loc[appliance, f"{level.lower()}_kwh"]) if known_appliance else context_pred
        daily_kwh = level_kwh_base * context_factor
        hours_per_day = None
    else:
        # hours_base already reflects the day-length cap applied once at
        # training time (see train_model.py) -- reused here as-is so the
        # number the user sees and the number the math uses are always
        # the same, and so a noisy power_max value can't produce an
        # impossible hours/day figure.
        hours_base = float(level_table.loc[appliance, f"{level.lower()}_hours"])
        hours_per_day = min(hours_base * context_factor, 23.5)

        aux_raw = meta.get("appliance_aux_raw", {})
        power_max_median = meta.get("power_max_median", 864.75)
        power_w = power_kw_override * 1000 if power_kw_override else aux_raw.get(appliance, {}).get("power_max", power_max_median)
        power_kw = max(power_w, 1.0) / 1000.0

        daily_kwh = power_kw * hours_per_day

    monthly_kwh = daily_kwh * days_per_month

    return {
        "predicted_kwh": round(monthly_kwh, 3),
        "daily_kwh": round(daily_kwh, 3),
        "hours_per_day": round(hours_per_day, 2) if hours_per_day is not None else None,
        "context_factor": round(context_factor, 3),
    }


def calculate_egyptian_tariff(kwh: float) -> float:
    if kwh <= 50:
        bill = kwh * 0.68
    elif kwh <= 100:
        bill = (50 * 0.68) + ((kwh - 50) * 0.78)
    elif kwh <= 200:
        bill = kwh * 0.95
    elif kwh <= 350:
        bill = (200 * 0.95) + ((kwh - 200) * 1.55)
    elif kwh <= 650:
        bill = (200 * 0.95) + (150 * 1.55) + ((kwh - 350) * 1.95)
    elif kwh <= 1000:
        bill = kwh * 2.10
    else:
        bill = kwh * 2.23
    return round(bill, 2)


def evaluate_budget(estimated_bill: float, user_budget: float, num_persons: int = 1):
    diff = estimated_bill - user_budget
    percentage = (estimated_bill / user_budget) * 100

    bill_per_person = round(estimated_bill / num_persons, 2)

    if percentage < 90.0:
        final_status = "Within Budget"
        message = f"Optimal consumption ({round(percentage, 1)}% of budget). Surplus: {abs(round(diff, 2))} EGP."
    elif 90.0 <= percentage <= 110.0:
        final_status = "Close to Budget"
        if diff <= 0:
            message = f"Warning: You are close to your budget limit ({round(percentage, 1)}% of budget). Remaining surplus: {abs(round(diff, 2))} EGP."
        else:
            message = f"Warning: Slightly over budget ({round(percentage, 1)}% of budget) by {round(diff, 2)} EGP."
    else:
        final_status = "Over Budget"
        message = f"Alert: Significantly over budget ({round(percentage, 1)}% of budget) by {round(diff, 2)} EGP!"

    return final_status, message, bill_per_person


def generate_recommendations(predicted_kwh, estimated_bill, user_budget, top_appliance="Fridge"):
    if not top_appliance:
        top_appliance = "Air Conditioner"

    percentage = (estimated_bill / user_budget) * 100
    recs = []

    if percentage >= 90.0:
        recs.append(f"It appears that the ({top_appliance}) is a major contributor to your high consumption.")
        recs.append("It is recommended to reduce unnecessary operating hours or adjust to energy-saving settings.")
        recs.append("Ensure appliances are not left on Standby Mode when not in use.")

        reduced_kwh = predicted_kwh * 0.85
        new_bill = calculate_egyptian_tariff(reduced_kwh)
        savings = estimated_bill - new_bill

        what_if = (
            f"If you optimize the usage of {top_appliance} and slightly reduce operation, "
            f"your bill will decrease to {new_bill} EGP (saving {round(savings, 2)} EGP)."
        )
    else:
        recs.append("Your energy consumption is well-optimized and within your safe budget limit.")
        recs.append("No immediate action is required. Keep maintaining your current usage patterns!")
        what_if = "Your budget is fully safe under current consumption levels. No optimized scenario needed."

    return recs, what_if


def run_smart_home_advisor(devices, household_size, user_budget, model, scaler_aux, meta, level_table):
    """
    devices: list of dicts, each {"appliance": str, "level": "Low"/"Medium"/"High", "power_kw": float|None}
    """
    device_results = []
    total_kwh = 0.0
    for d in devices:
        result = predict_device_kwh(
            d["appliance"], d["level"], household_size,
            model, scaler_aux, meta, level_table,
            power_kw_override=d.get("power_kw"),
        )
        device_results.append({**d, **result})
        total_kwh += result["predicted_kwh"]

    top_device = max(device_results, key=lambda d: d["predicted_kwh"]) if device_results else None
    top_appliance = top_device["appliance"] if top_device else None

    estimated_bill = calculate_egyptian_tariff(total_kwh)

    budget_status, budget_msg, cost_per_person = (None, None, None)
    recommendations, what_if = ([], None)
    if user_budget:
        budget_status, budget_msg, cost_per_person = evaluate_budget(estimated_bill, user_budget, household_size)
        recommendations, what_if = generate_recommendations(total_kwh, estimated_bill, user_budget, top_appliance)

    payload = {
        "best_model_name": meta["best_model_name"],
        "accuracy_r2": meta["best_model_r2"],
        "number_of_residents": household_size,
        "device_results": device_results,
        "predicted_kwh": round(total_kwh, 2),
        "estimated_bill": estimated_bill,
        "cost_per_person": cost_per_person,
        "budget_status": budget_status,
        "budget_message": budget_msg,
        "recommendations": recommendations,
        "what_if_scenario": what_if,
        "top_appliance": top_appliance,
    }
    return payload