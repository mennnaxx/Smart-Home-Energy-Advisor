import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


import pandas as pd

dataset1_path = "dataset1_smart_home_energy_consumption.csv"

df1 = pd.read_csv(dataset1_path)
df1_original = df1.copy()

df1.head()


print("Dataset shape:")
print(df1.shape)

print("\nDataset information:")
df1.info()

print("\nMissing values:")
print(df1.isnull().sum())

print("\nExact duplicate rows:")
print(df1.duplicated().sum())

print("Numerical variables summary:")
print(df1.describe())

print("\nAppliance types:")
print(df1["Appliance Type"].value_counts())

print("\nSeasons:")
print(df1["Season"].value_counts())

print("Energy Consumption statistics:")
print(df1["Energy Consumption (kWh)"].describe())

print("\nEnergy Consumption value counts:")
print(df1["Energy Consumption (kWh)"].value_counts().head(10))

plt.figure(figsize=(8, 5))
plt.hist(df1["Energy Consumption (kWh)"], bins=30)
plt.xlabel("Energy Consumption (kWh)")
plt.ylabel("Frequency")
plt.title("Distribution of Energy Consumption")
plt.show()

appliance_mean = (
    df1.groupby("Appliance Type")["Energy Consumption (kWh)"]
    .mean()
    .sort_values(ascending=False)
)
print("Average energy consumption by Appliance Type")
print(appliance_mean)

plt.figure(figsize=(10, 5))
plt.bar(appliance_mean.index, appliance_mean.values)
plt.xlabel("Appliance Type")
plt.ylabel("Average Energy Consumption (kWh)")
plt.title("Average Energy Consumption by Appliance Type")
plt.xticks(rotation=45)
plt.show()

season_mean = (
    df1.groupby("Season")["Energy Consumption (kWh)"]
    .mean()
)
print("Average energy consumption by season:")
print(season_mean)

plt.figure(figsize=(7, 5))
plt.bar(season_mean.index, season_mean.values)
plt.xlabel("Season")
plt.ylabel("Average Energy Consumption (kWh)")
plt.title("Average Energy Consumption by Season")
plt.show()

household_mean = (
    df1.groupby("Household Size")["Energy Consumption (kWh)"]
    .mean()
)

print("Average energy consumption by household size:")
print(household_mean)

plt.figure(figsize=(7, 5))
plt.bar(household_mean.index, household_mean.values)
plt.xlabel("Household Size")
plt.ylabel("Average Energy Consumption (kWh)")
plt.title("Average Energy Consumption by Household Size")
plt.show()

plt.figure(figsize=(8, 5))

plt.scatter(
    df1["Outdoor Temperature (°C)"].head(500),
    df1["Energy Consumption (kWh)"].head(500),
    s=20,
    alpha=0.7
)

plt.xlabel("Outdoor Temperature (°C)")
plt.ylabel("Energy Consumption (kWh)")
plt.title("Outdoor Temperature vs. Energy Consumption (First 100 Observations)")
plt.show()

correlation = df1["Outdoor Temperature (°C)"].corr(
    df1["Energy Consumption (kWh)"]
)

print("Correlation between temperature and energy consumption:")
print(correlation)

print("Unique time values:")
print(df1["Time"].unique())

print("\nNumber of unique time values:")
print(df1["Time"].nunique())


time_data = pd.to_datetime(df1["Time"], format="%H:%M")

hourly_mean = (
    df1.assign(Hour=time_data.dt.hour)
    .groupby("Hour")["Energy Consumption (kWh)"]
    .mean()
)

print("Average energy consumption by hour:")
print(hourly_mean)


plt.figure(figsize=(10, 5))

plt.plot(hourly_mean.index, hourly_mean.values, marker="o")

plt.xlabel("Hour of Day")
plt.ylabel("Average Energy Consumption (kWh)")
plt.title("Average Energy Consumption by Hour of Day")
plt.xticks(range(24))
plt.grid()
plt.show()


print("Earliest date:")
print(df1["Date"].min())

print("\nLatest date:")
print(df1["Date"].max())

print("\nNumber of unique dates:")
print(df1["Date"].nunique())


date_data = pd.to_datetime(df1["Date"])

monthly_mean = (
    df1.assign(Month=date_data.dt.month)
    .groupby("Month")["Energy Consumption (kWh)"]
    .mean()
)

print("Average energy consumption by month:")
print(monthly_mean)


plt.figure(figsize=(10, 5))

plt.plot(monthly_mean.index, monthly_mean.values, marker="o")

plt.xlabel("Month")
plt.ylabel("Average Energy Consumption (kWh)")
plt.title("Average Energy Consumption by Month")
plt.xticks(range(1, 13))
plt.grid()
plt.show()

plt.figure(figsize=(7, 5))
correlation_matrix = df1[
    ["Energy Consumption (kWh)", "Outdoor Temperature (°C)", "Household Size"]
].corr()

print("Correlation Matrix:")
print(correlation_matrix)
plt.imshow(correlation_matrix, cmap="coolwarm", vmin=-1, vmax=1)

plt.xticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns
)

plt.colorbar(label="Correlation")
plt.title("Correlation Matrix")

for i in range(len(correlation_matrix.columns)):
    for j in range(len(correlation_matrix.columns)):
        plt.text(
            j,
            i,
            f"{correlation_matrix.iloc[i, j]:.2f}",
            ha="center",
            va="center"
        )

plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 5))

plt.boxplot(df1["Energy Consumption (kWh)"] ,
    flierprops=dict(marker=".", markersize=2)
            )

plt.ylabel("Energy Consumption (kWh)")
plt.title("Box Plot of Energy Consumption")

plt.show()


Q1 = df1["Energy Consumption (kWh)"].quantile(0.25)
Q3 = df1["Energy Consumption (kWh)"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df1[
    (df1["Energy Consumption (kWh)"] < lower_bound) |
    (df1["Energy Consumption (kWh)"] > upper_bound)
]

print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)

print("\nLower bound:", lower_bound)
print("Upper bound:", upper_bound)

print("\nNumber of potential outliers:", len(outliers))
print("Percentage of potential outliers:", len(outliers) / len(df1) * 100)


appliance_stats = (
    df1.groupby("Appliance Type")["Energy Consumption (kWh)"]
    .agg(["mean", "max"])
    .sort_values("mean", ascending=False)
)

print("Energy consumption statistics by appliance:")
print(appliance_stats)


plt.figure(figsize=(12, 6))

df1.boxplot(
    column="Energy Consumption (kWh)",
    by="Appliance Type",
    rot=45
)

plt.xlabel("Appliance Type")
plt.ylabel("Energy Consumption (kWh)")
plt.title("Energy Consumption Distribution by Appliance Type")
plt.suptitle("")

plt.show()

dataset2_path = "dataset2_smart_home_energy_usage.csv"

df2 = pd.read_csv(dataset2_path)
df2_original = df2.copy()

df2.head()

print("Dataset shape:")
print(df2.shape)

print("\nDataset information:")
df2.info()

print("\nMissing values:")
print(df2.isnull().sum())

print("\nExact duplicate rows:")
print(df2.duplicated().sum())

print("Numerical variables summary:")
print(df2.describe())

print("\nOccupancy status:")
print(df2["occupancy_status"].value_counts())

print("\nAppliance categories:")
print(df2["appliance"].value_counts())

print("\nSeasons:")
print(df2["season"].value_counts())

print("\nDays of the week:")
print(df2["day_of_week"].value_counts())

print("\nHoliday values:")
print(df2["holiday"].value_counts())

print("Energy Consumption statistics:")
print(df2["energy_consumption_kWh"].describe())

print("\nEnergy Consumption value counts:")
print(
    df2["energy_consumption_kWh"]
    .value_counts()
    .head(10)
)

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))

plt.hist(
    df2["energy_consumption_kWh"],
    bins=30
)

plt.xlabel("Energy Consumption (kWh)")
plt.ylabel("Frequency")
plt.title("Distribution of Energy Consumption")

plt.show()

appliance_mean2 = (
    df2.groupby("appliance")["energy_consumption_kWh"]
    .mean()
    .sort_values(ascending=False)
)

print("Average energy consumption by appliance:")
print(appliance_mean2)

plt.figure(figsize=(10, 5))

plt.bar(
    appliance_mean2.index,
    appliance_mean2.values
)

plt.xlabel("Appliance Category")
plt.ylabel("Average Energy Consumption (kWh)")
plt.title("Average Energy Consumption by Appliance Category")
plt.xticks(rotation=45)

plt.show()

occupancy_mean = (
    df2.groupby("occupancy_status")["energy_consumption_kWh"]
    .mean()
)

print("Average energy consumption by occupancy status:")
print(occupancy_mean)

plt.figure(figsize=(7, 5))

plt.bar(
    occupancy_mean.index,
    occupancy_mean.values
)
plt.xlabel("Occupancy Status")
plt.ylabel("Average Energy Consumption (kWh)")
plt.title("Average Energy Consumption by Occupancy Status")

plt.show()

season_mean2 = (
    df2.groupby("season")["energy_consumption_kWh"]
    .mean()
)

print("Average energy consumption by season:")
print(season_mean2)

plt.figure(figsize=(7, 5))

plt.bar(
    season_mean2.index,
    season_mean2.values
)

plt.xlabel("Season")
plt.ylabel("Average Energy Consumption (kWh)")
plt.title("Average Energy Consumption by Season")

plt.show()

print("Usage duration statistics:")
print(df2["usage_duration_minutes"].describe())

plt.figure(figsize=(8, 5))

plt.hist(
    df2["usage_duration_minutes"],
    bins=100
)

plt.xlabel("Usage Duration (minutes)")
plt.ylabel("Frequency")
plt.title("Distribution of Appliance Usage Duration")

plt.show()

usage_by_appliance = (
    df2.groupby("appliance")["usage_duration_minutes"]
    .mean()
    .sort_values(ascending=False)
)

print("Average usage duration by appliance:")
print(usage_by_appliance)

plt.figure(figsize=(10, 5))

plt.bar(
    usage_by_appliance.index,
    usage_by_appliance.values
)

plt.xlabel("Appliance Category")
plt.ylabel("Average Usage Duration (minutes)")
plt.title("Average Usage Duration by Appliance Category")
plt.xticks(rotation=45)

plt.show()

duration_energy = (
    df2.groupby("usage_duration_minutes")["energy_consumption_kWh"]
    .mean()
)

print("Average energy consumption by usage duration:")
print(duration_energy.head(20))

print("Temperature setting statistics:")
print(df2["temperature_setting_C"].describe())

plt.figure(figsize=(8, 5))

plt.scatter(
    df2["temperature_setting_C"],
    df2["energy_consumption_kWh"],
    s=5,
    alpha=0.3
)

plt.xlabel("Temperature Setting (°C)")
plt.ylabel("Energy Consumption (kWh)")
plt.title("Temperature Setting vs. Energy Consumption")

plt.show()

temperature_correlation = (
    df2["temperature_setting_C"]
    .corr(df2["energy_consumption_kWh"])
)
print("Correlation between temperature setting and energy consumption:")
print(temperature_correlation)

timestamp_data = pd.to_datetime(
    df2["timestamp"]
)

print("Earliest timestamp:")
print(timestamp_data.min())

print("\nLatest timestamp:")
print(timestamp_data.max())

print("\nNumber of unique timestamps:")
print(timestamp_data.nunique())

hourly_mean2 = (
    df2.assign(Hour=timestamp_data.dt.hour)
    .groupby("Hour")["energy_consumption_kWh"]
    .mean()
)

print("Average energy consumption by hour:")
print(hourly_mean2)

plt.figure(figsize=(10, 5))

plt.plot(
    hourly_mean2.index,
    hourly_mean2.values,
    marker="o"
)

plt.xlabel("Hour of Day")
plt.ylabel("Average Energy Consumption (kWh)")
plt.title("Average Energy Consumption by Hour of Day")
plt.xticks(range(24))
plt.grid()

plt.show()

occupancy_usage = (
    df2.groupby("occupancy_status")["usage_duration_minutes"]
    .mean()
)

print("Average usage duration by occupancy status:")
print(occupancy_usage)

plt.figure(figsize=(7, 5))

plt.bar(
    occupancy_usage.index,
    occupancy_usage.values
)

plt.xlabel("Occupancy Status")
plt.ylabel("Average Usage Duration (minutes)")
plt.title("Average Usage Duration by Occupancy Status")

plt.show()

day_energy = (
    df2.groupby("day_of_week")["energy_consumption_kWh"]
    .mean()
)

print("Average energy consumption by day of week:")
print(day_energy)

plt.figure(figsize=(10, 5))

plt.bar(
    day_energy.index,
    day_energy.values
)

plt.xlabel("Day of Week")
plt.ylabel("Average Energy Consumption (kWh)")
plt.title("Average Energy Consumption by Day of Week")
plt.xticks(rotation=45)

plt.show()

holiday_energy = (
    df2.groupby("holiday")["energy_consumption_kWh"]
    .mean()
)

print("Average energy consumption by holiday status:")
print(holiday_energy)

plt.figure(figsize=(7, 5))

plt.bar(
    holiday_energy.index.astype(str),
    holiday_energy.values
)

plt.xlabel("Holiday")
plt.ylabel("Average Energy Consumption (kWh)")
plt.title("Average Energy Consumption by Holiday Status")

plt.show()

correlation_matrix2 = df2[
    [
        "energy_consumption_kWh",
        "temperature_setting_C",
        "usage_duration_minutes"
    ]
].corr()

print("Correlation Matrix:")
print(correlation_matrix2)


plt.figure(figsize=(7, 5))

plt.imshow(
    correlation_matrix2,
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)

plt.xticks(
    range(len(correlation_matrix2.columns)),
    correlation_matrix2.columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(correlation_matrix2.columns)),
    correlation_matrix2.columns
)

plt.colorbar(label="Correlation")

plt.title("Correlation Matrix")

for i in range(len(correlation_matrix2.columns)):
    for j in range(len(correlation_matrix2.columns)):
        plt.text(
            j,
            i,
            f"{correlation_matrix2.iloc[i, j]:.2f}",
            ha="center",
            va="center"
        )

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))

plt.boxplot(
    df2["energy_consumption_kWh"],
    flierprops=dict(
        marker=".",
        markersize=2
    )
)

plt.ylabel("Energy Consumption (kWh)")
plt.title("Box Plot of Energy Consumption")

plt.show()

Q1_2 = df2["energy_consumption_kWh"].quantile(0.25)
Q3_2 = df2["energy_consumption_kWh"].quantile(0.75)

IQR_2 = Q3_2 - Q1_2

lower_bound_2 = Q1_2 - 1.5 * IQR_2
upper_bound_2 = Q3_2 + 1.5 * IQR_2

outliers2 = df2[
    (df2["energy_consumption_kWh"] < lower_bound_2) |
    (df2["energy_consumption_kWh"] > upper_bound_2)
]

print("Q1:", Q1_2)
print("Q3:", Q3_2)
print("IQR:", IQR_2)

print("\nLower bound:", lower_bound_2)
print("Upper bound:", upper_bound_2)

print("\nNumber of potential outliers:", len(outliers2))

print(
    "Percentage of potential outliers:",
    len(outliers2) / len(df2) * 100
)

appliance_stats2 = (
    df2.groupby("appliance")["energy_consumption_kWh"]
    .agg(["mean", "max"])
    .sort_values("mean", ascending=False)
)

print("Energy consumption statistics by appliance:")
print(appliance_stats2)

plt.figure(figsize=(12, 6))

df2.boxplot(
    column="energy_consumption_kWh",
    by="appliance",
    rot=45
)

plt.xlabel("Appliance Category")
plt.ylabel("Energy Consumption (kWh)")
plt.title("Energy Consumption Distribution by Appliance Category")
plt.suptitle("")

plt.show()

dataset3_path = "dataset3_selected_appliances.csv"

df3 = pd.read_csv(dataset3_path)
df3_original = df3.copy()

print("First 5 rows:")
print(df3.head())

print("Dataset shape:")
print(df3.shape)

print("\nDataset information:")
df3.info()

print("\nMissing values:")
print(df3.isnull().sum())

print("\nExact duplicate rows:")
print(df3.duplicated().sum())

print("Column names:")
print(df3.columns.tolist())

print("\nData types:")
print(df3.dtypes)

print("\nStandardized appliance names:")
print(df3["standardized_appliance_name"].value_counts())

print("\nAppliance categories:")
print(df3["appliance_category"].value_counts())

appliance_counts = (
    df3["standardized_appliance_name"]
    .value_counts()
)

print("Number of devices by appliance:")
print(appliance_counts)

plt.figure(figsize=(12, 6))

plt.bar(
    appliance_counts.index,
    appliance_counts.values
)

plt.xlabel("Appliance Type")
plt.ylabel("Number of Devices")
plt.title("Number of Devices by Appliance Type")
plt.xticks(rotation=45)

plt.show()

category_counts = (
    df3["appliance_category"]
    .value_counts()
)

print("Number of devices by appliance category:")
print(category_counts)

plt.figure(figsize=(8, 5))

plt.bar(
    category_counts.index,
    category_counts.values
)

plt.xlabel("Appliance Category")
plt.ylabel("Number of Devices")
plt.title("Number of Devices by Appliance Category")
plt.xticks(rotation=45)

plt.show()

print("Available duration statistics:")
print(df3["available_duration"].describe())

plt.figure(figsize=(8, 5))

plt.hist(
    df3["available_duration"],
    bins=10
)

plt.xlabel("Available Duration (days)")
plt.ylabel("Frequency")
plt.title("Distribution of Available Duration")

plt.show()

print("Power maximum statistics:")
print(df3["power_max"].describe())

print("\nMinimum recorded power:")
print(df3["power_max"].min())

print("\nMaximum recorded power:")
print(df3["power_max"].max())

plt.figure(figsize=(8, 5))

plt.hist(
    df3["power_max"],
    bins=10
)

plt.xlabel("Maximum Recorded Power (W)")
plt.ylabel("Frequency")
plt.title("Distribution of Maximum Recorded Power")

plt.show()

appliance_power = (
    df3.groupby("standardized_appliance_name")["power_max"]
    .agg(["min", "mean", "max"])
    .sort_values("mean", ascending=False)
)

print("Power statistics by appliance:")
print(appliance_power)

appliance_power_mean = (
    df3.groupby("standardized_appliance_name")["power_max"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(12, 6))

plt.bar(
    appliance_power_mean.index,
    appliance_power_mean.values
)

plt.xlabel("Appliance Type")
plt.ylabel("Average Maximum Recorded Power (W)")
plt.title("Average Maximum Recorded Power by Appliance Type")
plt.xticks(rotation=45)

plt.show()

plt.figure(figsize=(12, 6))

df3.boxplot(
    column="power_max",
    by="standardized_appliance_name",
    rot=45
)

plt.xlabel("Appliance Type")
plt.ylabel("Maximum Recorded Power (W)")
plt.title("Power Distribution by Appliance Type")
plt.suptitle("")

plt.show()

availability_by_appliance = (
    df3.groupby("standardized_appliance_name")["available_duration"]
    .mean()
    .sort_values(ascending=False)
)

print("Average available duration by appliance:")
print(availability_by_appliance)

plt.figure(figsize=(12, 6))

plt.bar(
    availability_by_appliance.index,
    availability_by_appliance.values
)

plt.xlabel("Appliance Type")
plt.ylabel("Average Available Duration (days)")
plt.title("Average Available Duration by Appliance Type")
plt.xticks(rotation=45)

plt.show()

Q1_power = df3["power_max"].quantile(0.25)
Q3_power = df3["power_max"].quantile(0.75)

IQR_power = Q3_power - Q1_power

lower_power_bound = Q1_power - 1.5 * IQR_power
upper_power_bound = Q3_power + 1.5 * IQR_power

power_outliers = df3[
    (df3["power_max"] < lower_power_bound) |
    (df3["power_max"] > upper_power_bound)
]

print("Q1:", Q1_power)
print("Q3:", Q3_power)
print("IQR:", IQR_power)

print("\nLower bound:", lower_power_bound)
print("Upper bound:", upper_power_bound)

print("\nNumber of potential power outliers:")
print(len(power_outliers))

print("\nPercentage of potential power outliers:")
print(len(power_outliers) / len(df3) * 100)

print("Unique original plug names:")
print(df3["plug_name"].nunique())

print("\nUnique standardized appliance names:")
print(df3["standardized_appliance_name"].nunique())

print("\nUnique appliance categories:")
print(df3["appliance_category"].nunique())

appliance_device_counts = (
    df3.groupby("standardized_appliance_name")["id"]
    .count()
    .sort_values(ascending=False)
)

print("\nNumber of devices per standardized appliance:")
print(appliance_device_counts)

home_energy = (
    df1.groupby("Home ID")["Energy Consumption (kWh)"]
    .agg(["count", "mean", "sum"])
    .sort_values("mean", ascending=False)
)

print("Number of homes:", df1["Home ID"].nunique())
print("\nTop 10 homes by average energy consumption:")
print(home_energy.head(10))

plt.figure(figsize=(10, 5))
plt.hist(home_energy["mean"], bins=30)
plt.xlabel("Average Energy Consumption per Home (kWh)")
plt.ylabel("Number of Homes")
plt.title("Distribution of Average Energy Consumption Across Homes")
plt.show()

plt.figure(figsize=(10, 5))
plt.hist(home_energy["count"], bins=30)
plt.xlabel("Number of Records per Home")
plt.ylabel("Number of Homes")
plt.title("Distribution of Records per Home")
plt.show()

plt.figure(figsize=(12, 6))
for appliance in df1["Appliance Type"].dropna().unique():
    subset = df1[df1["Appliance Type"] == appliance]
    plt.scatter(
        subset["Outdoor Temperature (°C)"],
        subset["Energy Consumption (kWh)"],
        s=10,
        alpha=0.35,
        label=appliance
    )
plt.xlabel("Outdoor Temperature (°C)")
plt.ylabel("Energy Consumption (kWh)")
plt.title("Outdoor Temperature vs Energy Consumption by Appliance")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

time_data_extra = pd.to_datetime(df1["Time"], format="%H:%M")
hour_appliance = (
    df1.assign(Hour=time_data_extra.dt.hour)
    .groupby(["Hour", "Appliance Type"])["Energy Consumption (kWh)"]
    .mean()
    .unstack()
)

print("Average energy consumption by hour and appliance:")
print(hour_appliance)

plt.figure(figsize=(12, 6))
for appliance in hour_appliance.columns:
    plt.plot(hour_appliance.index, hour_appliance[appliance], marker="o", label=appliance)
plt.xlabel("Hour of Day")
plt.ylabel("Average Energy Consumption (kWh)")
plt.title("Hourly Energy Consumption by Appliance")
plt.xticks(range(24))
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid()
plt.tight_layout()
plt.show()

df2_time_extra = pd.to_datetime(df2["timestamp"])
df2_time_features = df2.assign(
    Month=df2_time_extra.dt.month,
    Weekday=df2_time_extra.dt.day_name(),
    Weekend=df2_time_extra.dt.dayofweek >= 5
)

monthly_energy2 = df2_time_features.groupby("Month")["energy_consumption_kWh"].mean()
weekday_energy2 = df2_time_features.groupby("Weekday")["energy_consumption_kWh"].mean()
weekend_energy2 = df2_time_features.groupby("Weekend")["energy_consumption_kWh"].mean()

print("Average energy by month:")
print(monthly_energy2)
print("\nAverage energy by weekday:")
print(weekday_energy2)
print("\nAverage energy by weekend status:")
print(weekend_energy2)

appliance_occupancy_energy = (
    df2.groupby(["appliance", "occupancy_status"])["energy_consumption_kWh"]
    .mean()
    .unstack()
)

print("Average energy by appliance and occupancy:")
print(appliance_occupancy_energy)

appliance_occupancy_energy.plot(kind="bar", figsize=(12, 6))
plt.xlabel("Appliance")
plt.ylabel("Average Energy Consumption (kWh)")
plt.title("Energy Consumption by Appliance and Occupancy Status")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.scatter(
    df2["usage_duration_minutes"],
    df2["energy_consumption_kWh"],
    s=8,
    alpha=0.25
)
plt.xlabel("Usage Duration (minutes)")
plt.ylabel("Energy Consumption (kWh)")
plt.title("Usage Duration vs Energy Consumption")
plt.grid()
plt.show()

print("Correlation between usage duration and energy:",
      df2["usage_duration_minutes"].corr(df2["energy_consumption_kWh"]))

df3_energy_potential = df3.assign(
    theoretical_energy_Wh=df3["power_max"] * df3["available_duration"] * 24
)

potential_by_appliance = (
    df3_energy_potential
    .groupby("standardized_appliance_name")["theoretical_energy_Wh"]
    .mean()
    .sort_values(ascending=False)
)

print("Theoretical energy potential by appliance (Wh):")
print(potential_by_appliance)

plt.figure(figsize=(12, 6))
plt.bar(potential_by_appliance.index, potential_by_appliance.values)
plt.xlabel("Appliance Type")
plt.ylabel("Theoretical Energy Potential (Wh)")
plt.title("Theoretical Energy Potential by Appliance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("=" * 70)
print("FINAL EDA INSIGHTS")
print("=" * 70)

print("\nDataset 1")
print("- Highest average-energy appliance:", appliance_mean.index[0], "->", round(appliance_mean.iloc[0], 3), "kWh")
print("- Highest average-energy season:", season_mean.idxmax(), "->", round(season_mean.max(), 3), "kWh")
print("- Highest average-energy household size:", household_mean.idxmax(), "->", round(household_mean.max(), 3), "kWh")
print("- Temperature-energy correlation:", round(df1["Outdoor Temperature (°C)"].corr(df1["Energy Consumption (kWh)"]), 4))
print("- Highest average-energy hour:", hourly_mean.idxmax(), "->", round(hourly_mean.max(), 3), "kWh")

print("\nDataset 2")
print("- Highest average-energy appliance:", appliance_mean2.index[0], "->", round(appliance_mean2.iloc[0], 3), "kWh")
print("- Highest average-energy occupancy status:", occupancy_mean.idxmax(), "->", round(occupancy_mean.max(), 3), "kWh")
print("- Highest average-energy season:", season_mean2.idxmax(), "->", round(season_mean2.max(), 3), "kWh")
print("- Temperature-setting correlation:", round(temperature_correlation, 4))
print("- Usage-duration/energy correlation:", round(df2["usage_duration_minutes"].corr(df2["energy_consumption_kWh"]), 4))
print("- Highest average-energy hour:", hourly_mean2.idxmax(), "->", round(hourly_mean2.max(), 3), "kWh")

print("\nDataset 3")
print("- Highest average-power appliance:", appliance_power_mean.index[0], "->", round(appliance_power_mean.iloc[0], 2), "W")
print("- Highest average-availability appliance:", availability_by_appliance.index[0], "->", round(availability_by_appliance.iloc[0], 2), "days")
print("- Number of standardized appliance types:", df3["standardized_appliance_name"].nunique())
print("- Number of appliance categories:", df3["appliance_category"].nunique())

print("\nThese findings can guide later feature engineering, prediction, clustering, or recommendation steps in the Smart Home Energy Advisor.")

print("Missing values in df1:")
print(df1.isnull().sum())

print("Duplicates before:", df1.duplicated().sum())
df1 = df1.drop_duplicates()
print("Duplicates after:", df1.duplicated().sum())
print("Shape:", df1.shape)

def iqr_bounds(series):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr

outlier_mask = pd.Series(False, index=df1.index)
for appliance, group in df1.groupby("Appliance Type")["Energy Consumption (kWh)"]:
    lower, upper = iqr_bounds(group)
    outlier_mask.loc[group.index] = (group < lower) | (group > upper)

print(f"Per-appliance outliers: {outlier_mask.sum()} ({outlier_mask.mean()*100:.2f}%)")

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

df1 = pd.get_dummies(
    df1,
    columns=["Appliance Type", "Season", "DayOfWeek"],
    drop_first=True
)

from sklearn.model_selection import RandomizedSearchCV, train_test_split

df1_model = df1.drop(columns=["Home ID"])

X1 = df1_model.drop(columns=["Energy Consumption (kWh)"])
y1 = df1_model["Energy Consumption (kWh)"]

X1_train, X1_test, y1_train, y1_test = train_test_split(
    X1, y1, test_size=0.2, random_state=42
)

print("Train shape:", X1_train.shape, "Test shape:", X1_test.shape)

from sklearn.preprocessing import StandardScaler

numeric_cols_1 = ["Outdoor Temperature (°C)", "Household Size", "Hour", "Month"]

scaler_1 = StandardScaler()
X1_train[numeric_cols_1] = scaler_1.fit_transform(X1_train[numeric_cols_1])
X1_test[numeric_cols_1] = scaler_1.transform(X1_test[numeric_cols_1])

print("Scaler was fit on X1_train only.")
print("Train mean after scaling (should be ~0):", X1_train[numeric_cols_1].mean().round(3).to_dict())
print("Test mean after scaling (not necessarily 0 - this proves no leakage):",
      X1_test[numeric_cols_1].mean().round(3).to_dict())

print("Missing values in df2:")
print(df2.isnull().sum())

print("Duplicates before:", df2.duplicated().sum())
df2 = df2.drop_duplicates()
print("Duplicates after:", df2.duplicated().sum())
print("Shape:", df2.shape)

def iqr_bounds(series):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr

outlier_mask2 = pd.Series(False, index=df2.index)
for appliance, group in df2.groupby("appliance")["energy_consumption_kWh"]:
    lower, upper = iqr_bounds(group)
    outlier_mask2.loc[group.index] = (group < lower) | (group > upper)

print(f"Per-appliance outliers: {outlier_mask2.sum()} ({outlier_mask2.mean()*100:.4f}%)")

ts = pd.to_datetime(df2["timestamp"])
df2["Hour"] = ts.dt.hour
df2["Month"] = ts.dt.month
df2["IsWeekend"] = (ts.dt.dayofweek >= 5).astype(int)

df2 = df2.drop(columns=["timestamp"])

df2["occupancy_status"] = df2["occupancy_status"].map(
    {"Occupied": 1, "Unoccupied": 0}
)

df2 = pd.get_dummies(
    df2,
    columns=["appliance", "season", "day_of_week"],
    drop_first=True
)

from sklearn.model_selection import train_test_split

df2_model = df2.drop(columns=["home_id"])

X2 = df2_model.drop(columns=["energy_consumption_kWh"])
y2 = df2_model["energy_consumption_kWh"]

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.2, random_state=42
)

print("Train shape:", X2_train.shape, "Test shape:", X2_test.shape)

from sklearn.preprocessing import StandardScaler

numeric_cols_2 = ["temperature_setting_C", "usage_duration_minutes", "Hour", "Month"]

scaler_2 = StandardScaler()
X2_train[numeric_cols_2] = scaler_2.fit_transform(X2_train[numeric_cols_2])
X2_test[numeric_cols_2] = scaler_2.transform(X2_test[numeric_cols_2])

print("Train mean after scaling (should be ~0):",
      X2_train[numeric_cols_2].mean().round(3).to_dict())
print("Test mean after scaling (not exactly 0 - confirms no leakage):",
      X2_test[numeric_cols_2].mean().round(3).to_dict())

print("Missing values in df3:")
print(df3.isnull().sum())

df3 = df3.drop(columns=["comment"])

print("Duplicates before:", df3.duplicated().sum())
df3 = df3.drop_duplicates()
print("Duplicates after:", df3.duplicated().sum())
print("Shape:", df3.shape)

def iqr_bounds(series):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr

for col in ["power_max", "available_duration"]:
    lower, upper = iqr_bounds(df3[col])
    n_out = ((df3[col] < lower) | (df3[col] > upper)).sum()
    print(f"{col}: {n_out} potential outliers")

df3 = df3.drop(columns=["id", "first_ts", "last_ts", "plug_name", "files_names"], errors='ignore')

df3 = pd.get_dummies(
    df3,
    columns=["standardized_appliance_name", "appliance_category"],
    drop_first=True
)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

numeric_cols_3 = ["power_max", "available_duration"]
scaler_3 = StandardScaler()
df3[numeric_cols_3] = scaler_3.fit_transform(df3[numeric_cols_3])


def get_models():
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=50, max_depth=10, n_jobs=-1, random_state=42
        ),
        "XGBoost": XGBRegressor(
            n_estimators=50, max_depth=5, learning_rate=0.1,
            objective="reg:squarederror", n_jobs=-1, random_state=42
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42
        ),
    }


def evaluate_models(models, X_train, y_train, X_test, y_test):
    results = {}
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results[name] = {
            "MAE": mean_absolute_error(y_test, preds),
            "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
            "R2": r2_score(y_test, preds),
        }
        fitted[name] = model
    return pd.DataFrame(results).T, fitted


results_df, fitted_models = evaluate_models(get_models(), X1_train, y1_train, X1_test, y1_test)
best_name = results_df["R2"].idxmax()
print(f"Best baseline model: {best_name} (R2 = {results_df.loc[best_name, 'R2']:.4f})")


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
aux_features["power_max"] = aux_features["power_max"].fillna(aux_features["power_max"].median())
aux_features["available_duration"] = aux_features["available_duration"].fillna(aux_features["available_duration"].median())

X1_augmented = X1.copy()
X1_augmented[["power_max", "available_duration", "Auxiliary_Data_Available"]] = aux_features[
    ["power_max", "available_duration", "Auxiliary_Data_Available"]
].values


X1_aug_train, X1_aug_test, y1_aug_train, y1_aug_test = train_test_split(
    X1_augmented, y1, test_size=0.2, random_state=42
)

aux_numeric_cols = ["power_max", "available_duration"]
scaler_aux = StandardScaler()
X1_aug_train[aux_numeric_cols] = scaler_aux.fit_transform(X1_aug_train[aux_numeric_cols])
X1_aug_test[aux_numeric_cols] = scaler_aux.transform(X1_aug_test[aux_numeric_cols])


augmented_results_df, fitted_augmented_models = evaluate_models(
    get_models(), X1_aug_train, y1_aug_train, X1_aug_test, y1_aug_test
)
print("Augmented Model Results:\n", augmented_results_df)


comparison_df = pd.DataFrame({
    "Baseline R2": results_df["R2"],
    "Augmented R2": augmented_results_df["R2"],
})
comparison_df["R2 Change"] = comparison_df["Augmented R2"] - comparison_df["Baseline R2"]
print("Baseline vs Augmented:\n", comparison_df)


final_model = fitted_augmented_models["Linear Regression"]
final_predictions = final_model.predict(X1_aug_test)

print("Final Model: Linear Regression")
print("MAE :", round(mean_absolute_error(y1_aug_test, final_predictions), 4))
print("RMSE:", round(np.sqrt(mean_squared_error(y1_aug_test, final_predictions)), 4))
print("R2  :", round(r2_score(y1_aug_test, final_predictions), 4))


plt.figure(figsize=(8, 6))
plt.scatter(y1_aug_test, final_predictions, alpha=0.3)
plt.plot([y1_aug_test.min(), y1_aug_test.max()], [y1_aug_test.min(), y1_aug_test.max()], linestyle="--")
plt.xlabel("Actual Energy Consumption (kWh)")
plt.ylabel("Predicted Energy Consumption (kWh)")
plt.title("Actual vs Predicted Energy Consumption")
plt.tight_layout()
plt.show()


residuals = y1_aug_test - final_predictions
plt.figure(figsize=(8, 6))
plt.scatter(final_predictions, residuals, alpha=0.3)
plt.axhline(y=0, linestyle="--")
plt.xlabel("Predicted Energy Consumption (kWh)")
plt.ylabel("Residual")
plt.title("Residual Analysis")
plt.tight_layout()
plt.show()


feature_coefficients = pd.DataFrame({
    "Feature": X1_aug_train.columns,
    "Coefficient": final_model.coef_,
})
feature_coefficients["Absolute_Coefficient"] = feature_coefficients["Coefficient"].abs()
feature_coefficients = feature_coefficients.sort_values("Absolute_Coefficient", ascending=False)
print("Most Influential Features:\n", feature_coefficients.head(15))

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.bar(augmented_results_df.index, augmented_results_df["R2"], color=['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728'])
plt.xlabel("Models")
plt.ylabel("R2 Score")
plt.title("Model Comparison (Step 4: Best Model Selection)")
plt.ylim(0, 1)

for i, v in enumerate(augmented_results_df["R2"]):
    plt.text(i, v + 0.02, f"{v:.4f}", ha='center', fontweight='bold')

plt.tight_layout()
plt.show()

best_model = final_model
sample_user_input = X1_aug_test.iloc[[0]]
predicted_kwh = float(best_model.predict(sample_user_input)[0])
print(f"Predicted Energy Consumption: {round(predicted_kwh, 2)} kWh")

def calculate_egyptian_tariff(kwh):
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

def evaluate_budget(estimated_bill, user_budget, num_persons=1):
    diff = estimated_bill - user_budget
    percentage = (estimated_bill / user_budget) * 100

    bill_per_person = round(estimated_bill / num_persons, 2)
    budget_per_person = round(user_budget / num_persons, 2)

    lower_bound_bill = user_budget * 0.90
    upper_bound_bill = user_budget * 1.10

    if estimated_bill < lower_bound_bill:
        bill_check = "Within Budget"
    elif lower_bound_bill <= estimated_bill <= upper_bound_bill:
        bill_check = "Close to Budget"
    else:
        bill_check = "Over Budget"

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

def run_smart_home_advisor(user_input_data, user_budget, num_persons=1, usage_level="Medium", custom_power_kw=0.0, top_appliance=None):

    best_model = final_model
    best_accuracy = round(r2_score(y1_aug_test, final_predictions), 4)

    if usage_level == "Low":
        daily_hours = 2
    elif usage_level == "High":
        daily_hours = 8
    else:
        daily_hours = 5

    if custom_power_kw and custom_power_kw > 0:
        predicted_kwh = custom_power_kw * daily_hours * 30
        calc_method = "Custom Appliance Power (Advanced Mode)"
    else:
        predicted_kwh = float(best_model.predict(user_input_data)[0]) * daily_hours * 30
        calc_method = "ML Model (Linear Regression)"

    estimated_bill = calculate_egyptian_tariff(predicted_kwh)
    budget_status, budget_msg, bill_per_person = evaluate_budget(estimated_bill, user_budget, num_persons)
    recommendations, what_if = generate_recommendations(predicted_kwh, estimated_bill, user_budget, top_appliance)

    payload = {
        "best_model_name": "Linear Regression",
        "accuracy_r2": best_accuracy,
        "calculation_method": calc_method,
        "number_of_residents": num_persons,
        "selected_usage_level": usage_level,
        "calculated_daily_hours": daily_hours,
        "custom_power_kw": custom_power_kw,
        "predicted_kwh": round(predicted_kwh, 2),
        "estimated_bill": estimated_bill,
        "cost_per_person": bill_per_person,
        "budget_status": budget_status,
        "budget_message": budget_msg,
        "recommendations": recommendations,
        "what_if_scenario": what_if
    }

    return payload