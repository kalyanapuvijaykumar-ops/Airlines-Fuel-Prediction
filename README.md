# Airline Fuel Prediction

This project predicts aircraft fuel burn using a cleaned airline fuel log dataset.

## Project structure

- `data/clean/flight_fuel_logs_clean.csv` – cleaned EDA-ready dataset
- `src/train_model.py` – trains the regression model
- `src/predict.py` – predicts fuel use for a new flight
- `models/flight_fuel_model.joblib` – trained model artifact
- `models/metrics.json` – model performance metrics

## Model choice

Based on the EDA, the relationship between fuel consumption and distance, payload, headwind, altitude, and aircraft type is strongly linear. A `LinearRegression` model was selected because it performs extremely well on this dataset and provides easy-to-interpret predictions.

## Train the model

```bash
python src/train_model.py
```

## Predict fuel usage

```bash
python src/predict.py --distance_km 1200 --payload_tonnes 10.5 --headwind_kts 8 --cruise_altitude_ft 33000 --aircraft_type B737
```

## Expected output

The script will print a prediction in kilograms, such as:

```text
Predicted fuel consumption: 5476.18 kg
```
