from pathlib import Path

import pandas as pd


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    raw_path = project_root / "data" / "raw" / "flight_fuel_logs.csv"
    clean_dir = project_root / "data" / "clean"
    clean_dir.mkdir(parents=True, exist_ok=True)
    output_path = clean_dir / "flight_fuel_logs_clean.csv"

    df = pd.read_csv(raw_path)

    # Normalize columns and clean values
    df.columns = [str(col).strip().lower() for col in df.columns]
    df = df.rename(columns={
        "distance_km": "distance_km",
        "payload_tonnes": "payload_tonnes",
        "headwind_kts": "headwind_kts",
        "cruise_altitude_ft": "cruise_altitude_ft",
        "aircraft_type": "aircraft_type",
        "fuel_kg": "fuel_kg",
    })

    for col in [
        "distance_km",
        "payload_tonnes",
        "headwind_kts",
        "cruise_altitude_ft",
        "fuel_kg",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["aircraft_type"] = df["aircraft_type"].astype(str).str.strip().str.upper()

    # Remove rows with any missing values and duplicates, then sort for EDA flow
    df = df.dropna().drop_duplicates().sort_values("distance_km").reset_index(drop=True)

    # Add a useful metric for exploratory analysis
    df["fuel_per_km"] = df["fuel_kg"] / df["distance_km"]

    # Save cleaned dataset
    df.to_csv(output_path, index=False)

    print(f"Cleaned dataset saved to: {output_path}")
    print(df.head().to_string(index=False))
    print(f"\nRows: {len(df)} | Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
