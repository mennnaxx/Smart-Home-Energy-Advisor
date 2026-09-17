# Smart Home Energy Advisor — Streamlit App

## Run it
```
pip install -r requirements.txt
streamlit run app.py
```

## Files
- `train_model.py` — run once (or whenever the source CSVs change) to retrain and
  save the model + lookup tables into `artifacts/`. The app never retrains itself.
- `advisor_logic.py` — prediction pipeline, Egyptian tariff calculator, budget
  evaluator, recommendations/what-if engine (ported from the notebook).
- `app.py` — the Streamlit dashboard.
- `artifacts/` — saved model (`best_model.joblib`), scaler, metadata (`meta.json`),
  and the per-appliance Low/Medium/High table (`appliance_levels.csv`).

## Known limitations (please read before treating this as production-ready)
1. **Device list**: the dropdown uses dataset1's 10 appliance categories
   (Air Conditioning, Computer, Dishwasher, Fridge, Heater, Lights, Microwave,
   Oven, TV, Washing Machine) — these are what the trained model actually knows
   how to predict for, since the model's one-hot encoding was built from
   dataset1. Dataset3's 17-device list (from the screenshot) is a *different*
   inventory used only to enrich 7 of those 10 appliances with power-rating data;
   it can't be used as the selectable list without retraining the model on a
   merged appliance taxonomy.
2. **Low/Medium/High table**: computed from dataset1's real per-appliance
   `Energy Consumption (kWh)` quantiles, not dataset3 — dataset3 has no usage-hours
   or consumption column, only a static `power_max` rating and an
   `available_duration` (days the smart plug was monitored, not daily usage hours).
3. **Reading-level granularity**: dataset1 is one reading per appliance, not a
   monthly total, so `predicted_kwh` is a per-reading estimate scaled by the
   selected usage level — not a calibrated monthly bill. Treat the bill/budget
   numbers as directional, not exact.
4. Context features the model needs but the GUI doesn't collect (outdoor
   temperature, season, day of week, hour) are filled with sensible defaults
   (today's date, the appliance's historical peak hour, the season's average
   temperature) inside `advisor_logic.build_context()`.
