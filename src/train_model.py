from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "clean" / "flight_fuel_logs_clean.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "flight_fuel_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"


def build_pipeline() -> Pipeline:
    numeric_features = [
        "distance_km",
        "payload_tonnes",
        "headwind_kts",
        "cruise_altitude_ft",
    ]
    categorical_features = ["aircraft_type"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", numeric_features),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )
    return pipeline


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["fuel_kg"])
    y = df["fuel_kg"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    metrics = {
        "r2_score": round(r2_score(y_test, predictions), 6),
        "mae": round(mean_absolute_error(y_test, predictions), 3),
        "rmse": round(mse ** 0.5, 3),
        "mse": round(mse, 3),
    }

    joblib.dump(pipeline, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Model training complete")
    print(json.dumps(metrics, indent=2))
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
