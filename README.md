# Smart Home Energy Advisor

A machine learning project for predicting household energy consumption and turning that prediction into a practical electricity-bill advisor for Egyptian households.

## Project Links

Live Streamlit App: https://smart-home-energy-advisor-user-friendly.streamlit.app/
Project Presentation: https://canva.link/rq90pt0v6y273sv


## Project Overview

This project analyzes three smart-home datasets, cleans and engineers features from them, trains and compares several regression models to predict energy consumption, and then wraps the best model in an end-to-end **Smart Home Energy Advisor** pipeline. The advisor converts a predicted consumption value into an estimated electricity bill (using the Egyptian tariff structure), checks it against a user-defined budget, and generates personalized savings recommendations.

## Quick Start

For users who want to run the project quickly:

```
# 1. Open the notebook in Google Colab (it mounts Google Drive for data access)
#    Smart_Home_Energy_final.ipynb

# 2. Update the dataset paths if needed
#    /content/drive/MyDrive/Smart_Home_Energy_Project/dataset1_smart_home_energy_consumption.csv
#    /content/drive/MyDrive/Smart_Home_Energy_Project/dataset2_smart_home_energy_usage.csv
#    /content/drive/MyDrive/Smart_Home_Energy_Project/dataset3_selected_appliances.csv

# 3. Run all cells in order — EDA → Preprocessing → Modeling → Advisor
```

The project uses the following main technologies and libraries:

- **Python** — Core programming language
- **Pandas** — Data loading, cleaning, and manipulation
- **NumPy** — Numerical computations
- **Matplotlib** — Data visualization
- **Scikit-learn** — Preprocessing, train/test splitting, model training, and evaluation
- **XGBoost** — Gradient boosting regressor
- **Google Colab (google.colab.drive)** — Dataset access via Google Drive
- **Jupyter Notebook** — Experimentation and model development

## Project Structure

```
Smart_Home_Energy_Project/
│
├── dataset1_smart_home_energy_consumption.csv   # Appliance-level readings per home
├── dataset2_smart_home_energy_usage.csv         # Usage duration & occupancy behavior
├── dataset3_selected_appliances.csv             # Appliance metadata (power, availability)
|── Model Demo.mp4
|── smart_home_energy_advisor_final.pdf 
|── Smart Home Energy Advisor.pptx 
|
└── Smart_Home_Energy_final.ipynb                # Full EDA, preprocessing, modeling & advisor

```

## Datasets

The notebook combines three complementary datasets:

**Dataset 1 — Smart Home Energy Consumption**
Per-home, per-appliance energy readings with outdoor temperature, household size, season, date, and time. This is the main dataset used for training the prediction model.

**Dataset 2 — Smart Home Energy Usage**
Appliance-level usage duration, occupancy status, temperature setting, day of week, and holiday flag. Used as a supporting dataset to compare consumption patterns and behavior.

**Dataset 3 — Household Appliances (Metadata)**
Standardized appliance names, categories, maximum recorded power, and available duration. Used to augment Dataset 1 with appliance-level context (power and availability) as auxiliary features.

## Exploratory Data Analysis (EDA)

Each dataset is examined for:

- Shape, data types, missing values, and duplicate records
- Summary statistics for numerical variables
- Distribution of energy consumption (histograms, box plots)
- Average consumption by appliance type, season, household size, occupancy status, and holiday status
- Time-of-day and monthly consumption patterns
- Correlation between energy consumption and temperature / usage duration
- Outlier detection using the IQR method (overall and per-appliance)

Deeper relationship analysis is then added on top of the base EDA:

- Home-level energy analysis (average and total consumption per home)
- Temperature × appliance and hour × appliance interaction plots
- Timestamp-derived features (month, weekday, weekend) vs. energy usage
- Appliance × occupancy consumption comparison
- Usage duration vs. energy consumption scatter analysis
- A theoretical energy-potential estimate from Dataset 3 (`power_max × available_duration`)

A final summary block prints the key data-driven insights from all three datasets side by side.

## Preprocessing & Feature Engineering

The workflow below is applied independently to each dataset:

```
Handle Missing Values
       ↓
Remove Duplicates
       ↓
Cap Outliers (per-appliance IQR bounds)
       ↓
Feature Engineering (Hour, Month, DayOfWeek, IsWeekend)
       ↓
Categorical Encoding (One-Hot Encoding)
       ↓
Train/Test Split
       ↓
Feature Scaling (StandardScaler, fit on train only)
       ↓
Leak-Check (compare train vs. test means after scaling)
```

- **Outlier handling:** outliers are capped (not dropped) using IQR bounds computed **per appliance**, preserving legitimate high-usage appliances like air conditioners.
- **Time features:** `Hour`, `Month`, `DayOfWeek`, and `IsWeekend` are derived from the raw date/time columns, which are then dropped.
- **Encoding:** categorical columns (appliance type, season, day of week, occupancy status, appliance category) are one-hot encoded with `drop_first=True`.
- **Scaling:** `StandardScaler` is fit only on the training split and applied to both train and test sets to prevent data leakage; train/test means are printed as a sanity check.

## Model Training & Evaluation

Four regression models are trained and compared on Dataset 1:

| Model | Key Parameters |
|---|---|
| Linear Regression | default |
| Random Forest Regressor | `n_estimators=50`, `max_depth=10` |
| XGBoost Regressor | `n_estimators=50`, `max_depth=5`, `learning_rate=0.1` |
| Gradient Boosting Regressor | `n_estimators=50`, `max_depth=3`, `learning_rate=0.1` |

Each model is evaluated using:

- **MAE** — Mean Absolute Error
- **RMSE** — Root Mean Squared Error
- **R² Score** — proportion of variance explained

### Baseline vs. Augmented Comparison

Models are first trained as a **baseline** on Dataset 1 alone, then retrained on an **augmented** feature set that merges in appliance power and availability metadata from Dataset 3 (with a flag indicating whether auxiliary data was found for that appliance). Baseline and augmented R² scores are compared side by side to measure the added value of the metadata.

### Best Model Selection

**Linear Regression** on the augmented feature set is selected as the final model. It is evaluated with:

- MAE, RMSE, and R² on the held-out test set
- An Actual vs. Predicted scatter plot with a reference diagonal
- A residual plot to check for bias/heteroscedasticity
- A ranked table of the most influential features by absolute coefficient value

## Smart Home Energy Advisor

The trained model powers an end-to-end advisor pipeline:

**1. Energy Prediction**
The final model predicts hourly energy consumption (kWh), which is scaled up to a full monthly estimate (`hourly_kwh × 24 × 30`).

**2. Egyptian Electricity Tariff Calculator**
The predicted monthly consumption is converted into an estimated bill (EGP) using the tiered Egyptian residential tariff structure.

**3. Budget Evaluator**
The estimated bill is compared against a user-defined monthly budget, returning a status (*Within Budget*, *Close to Budget*, or *Over Budget*), a message, and the cost per household member.

**4. Recommendations & What-If Scenario Engine**
If consumption is near or over budget, the advisor flags the top contributing appliance, suggests concrete savings actions, and simulates a "what-if" scenario showing the bill after a 15% usage reduction.

**5. Master Pipeline**
`run_smart_home_advisor()` ties all of the above together — from raw user input to a single JSON-style payload containing the predicted usage, estimated bill, budget status, and recommendations.

```
User Input
    ↓
Trained Model → Predicted Hourly kWh
    ↓
Scale to Monthly kWh
    ↓
Egyptian Tariff Calculator → Estimated Bill
    ↓
Budget Evaluator → Budget Status + Cost per Person
    ↓
Recommendation Engine → Tips + What-If Scenario
    ↓
Final Advisor Payload
```

## Important Notes

- The datasets are loaded from Google Drive via `google.colab.drive`; running outside Colab requires adjusting the file paths.
- Outliers are **capped**, not removed, to preserve realistic high-consumption records per appliance.
- Feature scalers are fit on the training data only, and train/test means are printed to confirm no data leakage occurred.
- The augmented model combines Dataset 1 (energy readings) with Dataset 3 (appliance metadata) via an appliance-name mapping; missing metadata is filled with the median and flagged with `Auxiliary_Data_Available`.
- The Egyptian tariff calculator uses fixed tiered rates and should be updated if official tariffs change.

## Future Improvements

- Expand hyperparameter tuning (e.g., GridSearchCV/RandomizedSearchCV) across all four models.
- Add cross-validation for more robust model comparison.
- Incorporate additional real-time features such as live weather data.
- Extend the appliance-metadata mapping to cover more appliance types.
- Package the advisor pipeline into a simple web app (e.g., Streamlit) for interactive use.
- Add automated tests for the tariff calculator and budget evaluator functions.

## 📄 License

This project was developed for educational and research purposes.
