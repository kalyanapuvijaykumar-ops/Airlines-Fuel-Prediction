import argparse
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "flight_fuel_model.joblib"


def load_model():
    return joblib.load(MODEL_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict fuel consumption for an airline flight.")
    parser.add_argument("--distance_km", type=float, required=True)
    parser.add_argument("--payload_tonnes", type=float, required=True)
    parser.add_argument("--headwind_kts", type=float, required=True)
    parser.add_argument("--cruise_altitude_ft", type=int, required=True)
    parser.add_argument("--aircraft_type", type=str, required=True)

    args = parser.parse_args()

    sample = pd.DataFrame([
        {
            "distance_km": args.distance_km,
            "payload_tonnes": args.payload_tonnes,
            "headwind_kts": args.headwind_kts,
            "cruise_altitude_ft": args.cruise_altitude_ft,
            "aircraft_type": args.aircraft_type.upper(),
        }
    ])

    model = load_model()
    prediction = model.predict(sample)[0]
    print(f"Predicted fuel consumption: {prediction:.2f} kg")


if __name__ == "__main__":
    main()
