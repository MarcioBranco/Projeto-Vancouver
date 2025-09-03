#!/usr/bin/env python
import argparse
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import requests
from pydantic import BaseModel, ValidationError, conint, constr
from dotenv import load_dotenv
import os

# Setup logging
LOGS_DIR = Path("artifacts/logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=LOGS_DIR / "features.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

# Load environment variables
load_dotenv()
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# Constants
ROOT = Path.home() / "proj-van"
DATA_PROCESSED = ROOT / "data_processed"
FEATURES_DIR = DATA_PROCESSED / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

# Pydantic model for validation
class FeatureRow(BaseModel):
    year: conint(ge=2000, le=2100)  # type: ignore
    week: conint(ge=1, le=53)       # type: ignore
    category: constr(min_length=1)  # type: ignore
    neighbourhood: constr(min_length=1)  # type: ignore
    volume: conint(gt=0)            # type: ignore

def load_normalized(input_path: Path) -> pd.DataFrame:
    logging.info(f"Loading normalized data from {input_path}")
    return pd.read_parquet(input_path)

def engineer_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["opened_date"] = pd.to_datetime(df["opened_date"], errors="coerce", utc=True)
    df = df.dropna(subset=["opened_date"])
    df["week"] = df["opened_date"].dt.isocalendar().week
    df["year"] = df["opened_date"].dt.year
    df["month"] = df["opened_date"].dt.month
    df["dayofweek"] = df["opened_date"].dt.dayofweek
    return df

def aggregate_volumes(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["year", "week", "category", "neighbourhood"])
        .size()
        .reset_index(name="volume")
    )
    return grouped

def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["category", "neighbourhood", "year", "week"])
    df["moving_avg_4w"] = (
        df.groupby(["category", "neighbourhood"])["volume"]
        .transform(lambda x: x.rolling(4, min_periods=1).mean())
    )
    return df

def fetch_weather_data(year: int, week: int) -> dict:
    # OpenWeather API: only supports daily/hourly, so we fetch for each week (Mon-Sun)
    # We'll use Vancouver: lat=49.246, lon=-123.116
    try:
        # Find Monday of the ISO week
        monday = datetime.strptime(f"{year}-W{int(week):02d}-1", "%G-W%V-%u")
        sunday = monday + pd.Timedelta(days=6)
        url = (
            f"https://api.openweathermap.org/data/2.5/onecall/timemachine"
            f"?lat=49.246&lon=-123.116&dt={int(monday.timestamp())}"
            f"&units=metric&appid={OPENWEATHER_API_KEY}"
        )
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            logging.warning(f"Weather API failed for {year}-W{week}: {resp.status_code}")
            return {"rain_mm": 0.0, "temp_c": 0.0}
        data = resp.json()
        temps = [h["temp"] for h in data.get("hourly", []) if "temp" in h]
        rains = [h.get("rain", {}).get("1h", 0.0) for h in data.get("hourly", [])]
        avg_temp = np.mean(temps) if temps else 0.0
        total_rain = np.sum(rains) if rains else 0.0
        return {"rain_mm": float(total_rain), "temp_c": float(avg_temp)}
    except Exception as e:
        logging.error(f"Weather API error for {year}-W{week}: {e}")
        return {"rain_mm": 0.0, "temp_c": 0.0}

def add_external_features(df: pd.DataFrame) -> pd.DataFrame:
    if not OPENWEATHER_API_KEY:
        logging.warning("OPENWEATHER_API_KEY not set, filling weather with 0.")
        df["rain_mm"] = 0.0
        df["temp_c"] = 0.0
        return df

    weather_records = []
    unique_weeks = df[["year", "week"]].drop_duplicates()
    for _, row in unique_weeks.iterrows():
        year, week = int(row["year"]), int(row["week"])
        weather = fetch_weather_data(year, week)
        weather_records.append({"year": year, "week": week, **weather})
    weather_df = pd.DataFrame(weather_records)
    df = df.merge(weather_df, on=["year", "week"], how="left")
    df["rain_mm"] = df["rain_mm"].fillna(0.0)
    df["temp_c"] = df["temp_c"].fillna(0.0)
    return df

def detect_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["zscore"] = (
        df.groupby(["category", "neighbourhood"])["volume"]
        .transform(lambda x: (x - x.mean()) / (x.std(ddof=0) if x.std(ddof=0) > 0 else 1))
    )
    df["is_outlier"] = df["zscore"].abs() > 3
    df = df.drop(columns=["zscore"])
    return df

def validate_data(df: pd.DataFrame) -> None:
    errors = []
    for idx, row in df.iterrows():
        try:
            FeatureRow(
                year=int(row["year"]),
                week=int(row["week"]),
                category=str(row["category"]),
                neighbourhood=str(row["neighbourhood"]),
                volume=int(row["volume"])
            )
        except ValidationError as e:
            errors.append((idx, e))
    if errors:
        logging.error(f"Validation errors: {errors}")
        raise ValueError(f"Validation failed for {len(errors)} rows.")
    assert (df["volume"] > 0).all(), "All volumes must be > 0"

def save_features(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logging.info(f"Features saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Feature engineering for Vancouver 311")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input normalized parquet file (e.g., data_processed/normalized/normalized_requests.parquet)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output features parquet file (e.g., data_processed/features/features.parquet)"
    )
    args = parser.parse_args()

    try:
        df = load_normalized(Path(args.input))
        df = engineer_dates(df)
        df = aggregate_volumes(df)
        df = add_moving_averages(df)
        df = add_external_features(df)
        df = detect_outliers(df)
        validate_data(df)
        save_features(df, Path(args.output))
        logging.info("Feature engineering completed successfully.")
    except Exception as e:
        logging.exception(f"Feature engineering failed: {e}")
        raise

if __name__ == "__main__":
    main()
