"""
Training script for the Smart Home Energy Advisor.
Reproduces the preprocessing + modeling pipeline from the notebook
(Smart_Home_Energy_befpore_gui.ipynb) and saves everything the
Streamlit app needs so training never has to run again.
"""
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

DATA_DIR = "."
OUT_DIR = "./artifacts"
import os
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------
# 1) Load raw data
# ---------------------------------------------------------------
df1 = pd.read_csv(f"{DATA_DIR}/dataset1_smart_home_energy_consumption.csv")
df1_original = df1.copy()

df3 = pd.read_csv(f"{DATA_DIR}/dataset3_selected_appliances.csv")
df3_original = df3.copy()

# ---------------------------------------------------------------
# 2) Preprocess dataset1 (same steps as the notebook)
# ---------------------------------------------------------------
df1 = df1.drop_duplicates()

def iqr_bounds(series):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr

def cap(group):
    lower, upper = iqr_bounds(group)
    return group.clip(lower, upper)

df1["Energy Consumption (kWh)"] = (
    df1.groupby("Appliance Type")["Energy Consumption (kWh)"].transform(cap)
)

df1["Hour"] = pd.to_datetime(df1["Time"], format="%H:%M").dt.hour
date_parsed = pd.to_datetime(df1["Date"])
df1["Month"] = date_parsed.dt.month
df1["DayOfWeek"] = date_parsed.dt.day_name()
df1["IsWeekend"] = (date_parsed.dt.dayofweek >= 5).astype(int)
df1 = df1.drop(columns=["Time", "Date"])

APPLIANCE_TYPES = sorted(df1["Appliance Type"].unique().tolist())
SEASONS = sorted(df1["Season"].unique().tolist())
DAYS = sorted(df1["DayOfWeek"].unique().tolist())

df1_enc = pd.get_dummies(
    df1, columns=["Appliance Type", "Season", "DayOfWeek"], drop_first=True
)

df1_model = df1_enc.drop(columns=["Home ID"])
X1 = df1_model.drop(columns=["Energy Consumption (kWh)"])
y1 = df1_model["Energy Consumption (kWh)"]

# ---------------------------------------------------------------
# 3) Build auxiliary (dataset3) features, same mapping as notebook
# ---------------------------------------------------------------
appliance_mapping = {
    "Fridge": "fridge",
    "Dishwasher": "dishwasher",
    "Microwave": "micro_wave_oven",
    "Air Conditioning": "air_conditioner",
    "Computer": "computer",
    "TV": "tv",
    "Washing Machine": "washing_machine",
}

df1_aux = df1_original.copy()
df1_aux["appliance_metadata"] = df1_aux["Appliance Type"].map(appliance_mapping)

df3_aux = (
    df3_original[["standardized_appliance_name", "appliance_category", "power_max", "available_duration"]]
    .groupby("standardized_appliance_name", as_index=False)
    .agg({"power_max": "mean", "available_duration": "mean"})
    .rename(columns={"standardized_appliance_name": "appliance_metadata"})
)

aux_features = df1_aux[["appliance_metadata"]].merge(df3_aux, on="appliance_metadata", how="left")
aux_features["Auxiliary_Data_Available"] = aux_features["power_max"].notna().astype(int)
power_max_median = aux_features["power_max"].median()
avail_dur_median = aux_features["available_duration"].median()
aux_features["power_max"] = aux_features["power_max"].fillna(power_max_median)
aux_features["available_duration"] = aux_features["available_duration"].fillna(avail_dur_median)

X1_augmented = X1.copy()
X1_augmented[["power_max", "available_duration", "Auxiliary_Data_Available"]] = aux_features[
    ["power_max", "available_duration", "Auxiliary_Data_Available"]
].values

# ---------------------------------------------------------------
# 4) Train / test split + scale aux numeric cols (matches notebook)
# ---------------------------------------------------------------
X1_aug_train, X1_aug_test, y1_aug_train, y1_aug_test = train_test_split(
    X1_augmented, y1, test_size=0.2, random_state=42
)

aux_numeric_cols = ["power_max", "available_duration"]
scaler_aux = StandardScaler()
X1_aug_train[aux_numeric_cols] = scaler_aux.fit_transform(X1_aug_train[aux_numeric_cols])
X1_aug_test[aux_numeric_cols] = scaler_aux.transform(X1_aug_test[aux_numeric_cols])

FEATURE_COLUMNS = list(X1_aug_train.columns)

# ---------------------------------------------------------------
# 5) Train candidate models for comparison, but the production model
#    is now always Linear Regression (matches the updated notebook,
#    which reuses the fitted Linear Regression as `final_model`
#    instead of dynamically picking the best R2 among the four).
# ---------------------------------------------------------------
def get_models():
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=50, max_depth=5, learning_rate=0.1,
                                 objective="reg:squarederror", n_jobs=-1, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42),
    }

results = {}
fitted = {}
for name, model in get_models().items():
    model.fit(X1_aug_train, y1_aug_train)
    preds = model.predict(X1_aug_test)
    results[name] = {
        "MAE": mean_absolute_error(y1_aug_test, preds),
        "RMSE": np.sqrt(mean_squared_error(y1_aug_test, preds)),
        "R2": r2_score(y1_aug_test, preds),
    }
    fitted[name] = model

results_df = pd.DataFrame(results).T

# Final model is fixed to Linear Regression (notebook change), not the
# best-of-4 by R2 anymore. Kept name "best_*" for compatibility with the
# rest of the pipeline / meta.json schema.
best_name = "Linear Regression"
best_model = fitted[best_name]
best_r2 = float(results_df.loc[best_name, "R2"])

print("Model comparison:\n", results_df)
print(f"\nProduction model (fixed): {best_name} (R2={best_r2:.4f})")

# ---------------------------------------------------------------
# 6) Per-appliance Low/Medium/High consumption table
#    (built from dataset1's real Energy Consumption values,
#    NOT dataset3 -- dataset3 has no consumption/usage-hours data,
#    only a static power_max rating and a monitoring-period length)
#
#    NOTE: "low_kwh / medium_kwh / high_kwh" are the real per-appliance
#    quantile means from dataset1 -- i.e. what a Low/Medium/High user of
#    that specific appliance actually consumes, taken directly from the
#    data. This is NOT a fixed hour table (no hardcoded 2h/5h/8h for
#    every appliance) and NOT an invented ratio -- it's appliance-
#    specific and data-driven, per Low/Medium/High level.
# ---------------------------------------------------------------
level_table_rows = []
for appliance, group in df1.groupby("Appliance Type")["Energy Consumption (kWh)"]:
    p33, p66 = group.quantile([0.33, 0.66])
    low_mean = group[group <= p33].mean()
    med_mean = group[(group > p33) & (group <= p66)].mean()
    high_mean = group[group > p66].mean()
    overall_mean = group.mean()
    level_table_rows.append({
        "Appliance Type": appliance,
        "low_kwh": low_mean, "medium_kwh": med_mean, "high_kwh": high_mean,
        "overall_mean_kwh": overall_mean,
    })
level_table = pd.DataFrame(level_table_rows).set_index("Appliance Type")
print("\nPer-appliance Low/Medium/High consumption (kWh/day, real data quantiles):\n", level_table.round(3))

# Appliances that run continuously with a duty cycle (compressor on/off)
# rather than being switched on for a block of hours. It is not physically
# meaningful to show "High = 8 hours/day" for a fridge, so these are shown
# to the user as a consumption *level* (kept in kWh terms), never converted
# to an hours/day figure. Add more appliances here if the same logic
# applies to them (e.g. always-on routers, always-on freezers, etc.).
ALWAYS_ON_APPLIANCES = ["Fridge"]

# ---------------------------------------------------------------
# 7) Default context values (Outdoor Temp per season, peak Hour per
#    appliance, Household Size range) -- used to fill the model's
#    required features that the GUI does not ask the user for.
# ---------------------------------------------------------------
season_temp_avg = df1.groupby("Season")["Outdoor Temperature (°C)"].mean().to_dict()
appliance_peak_hour = (
    df1.groupby(["Appliance Type", "Hour"])["Energy Consumption (kWh)"]
    .mean().reset_index()
    .sort_values("Energy Consumption (kWh)", ascending=False)
    .drop_duplicates("Appliance Type")
    .set_index("Appliance Type")["Hour"].to_dict()
)
household_size_range = [int(df1["Household Size"].min()), int(df1["Household Size"].max())]

# Raw (unscaled) per-appliance aux values, so the app can build a
# feature row for any appliance without needing df3 at runtime.
appliance_aux_raw = (
    df1_aux[["Appliance Type", "appliance_metadata"]]
    .drop_duplicates()
    .merge(df3_aux, on="appliance_metadata", how="left")
    .assign(
        Auxiliary_Data_Available=lambda d: d["power_max"].notna().astype(int),
        power_max=lambda d: d["power_max"].fillna(power_max_median),
        available_duration=lambda d: d["available_duration"].fillna(avail_dur_median),
    )
    .set_index("Appliance Type")[["power_max", "available_duration", "Auxiliary_Data_Available"]]
    .to_dict(orient="index")
)

# ---------------------------------------------------------------
# 8) Real hours/day per appliance/level, derived the physical way:
#       hours/day = kWh_for_level (real data) / power_kW (real data)
#    This is the SAME number the app shows the user in the Low/Medium/
#    High dropdown -- computed once here, at training time, and saved,
#    instead of being recomputed separately in the app in a way that
#    never fed back into the actual monthly kWh calculation.
#
#    A few appliances only have a rough/default power rating (no real
#    dataset3 match), which can push the raw hour estimate past a
#    realistic day; when that happens all three levels for that
#    appliance are scaled down together (their relative proportions
#    are preserved) so the range stays under `MAX_DAILY_HOURS`.
# ---------------------------------------------------------------
MAX_DAILY_HOURS = 18.0

for appliance in level_table.index:
    aux = appliance_aux_raw.get(appliance, {"power_max": power_max_median})
    power_kw = max(aux["power_max"], 1.0) / 1000.0

    low_h = level_table.loc[appliance, "low_kwh"] / power_kw
    med_h = level_table.loc[appliance, "medium_kwh"] / power_kw
    high_h = level_table.loc[appliance, "high_kwh"] / power_kw

    if high_h > MAX_DAILY_HOURS:
        factor = MAX_DAILY_HOURS / high_h
        low_h, med_h, high_h = low_h * factor, med_h * factor, high_h * factor

    low_h = max(low_h, 0.15)
    med_h = max(med_h, low_h + 0.15)
    high_h = max(high_h, med_h + 0.15)

    level_table.loc[appliance, "low_hours"] = low_h
    level_table.loc[appliance, "medium_hours"] = med_h
    level_table.loc[appliance, "high_hours"] = high_h

print("\nPer-appliance Low/Medium/High hours/day (derived from real kWh and real power):\n",
      level_table[["low_hours", "medium_hours", "high_hours"]].round(2))

# ---------------------------------------------------------------
# 9) Save everything
# ---------------------------------------------------------------
joblib.dump(best_model, f"{OUT_DIR}/best_model.joblib")
joblib.dump(scaler_aux, f"{OUT_DIR}/scaler_aux.joblib")

meta = {
    "feature_columns": FEATURE_COLUMNS,
    "appliance_types": APPLIANCE_TYPES,
    "seasons": SEASONS,
    "days": DAYS,
    "appliance_mapping": appliance_mapping,
    "power_max_median": power_max_median,
    "available_duration_median": avail_dur_median,
    "best_model_name": best_name,
    "best_model_r2": best_r2,
    "model_comparison": results_df.round(4).to_dict(orient="index"),
    "season_temp_avg": season_temp_avg,
    "appliance_peak_hour": appliance_peak_hour,
    "household_size_range": household_size_range,
    "appliance_aux_raw": appliance_aux_raw,
    "always_on_appliances": ALWAYS_ON_APPLIANCES,
    "days_per_month": 30,
}
with open(f"{OUT_DIR}/meta.json", "w") as f:
    json.dump(meta, f, indent=2)

level_table.to_csv(f"{OUT_DIR}/appliance_levels.csv")

print("\nSaved model + metadata to", OUT_DIR)