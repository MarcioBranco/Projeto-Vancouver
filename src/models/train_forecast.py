#!/usr/bin/env python
import argparse
import logging
from pathlib import Path
import pandas as pd
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import joblib
from pydantic import BaseModel, ValidationError
import numpy as np

# Paths
ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = ROOT / "data_processed" / "features"
ARTIFACTS_MODELS = ROOT / "artifacts/models"
ARTIFACTS_LOGS = ROOT / "artifacts/logs"
ARTIFACTS_MODELS.mkdir(parents=True, exist_ok=True)
ARTIFACTS_LOGS.mkdir(parents=True, exist_ok=True)

# Logging
logging.basicConfig(
    filename=ARTIFACTS_LOGS / "models.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

class FeatureSchema(BaseModel):
    year: int
    week: int
    service_category: str
    neighborhood: str
    volume: int
    moving_avg_4w: float
    rain: float
    dayofweek: int

def load_features(input_path):
    df = pd.read_parquet(input_path)
    # Validate schema
    for _, row in df.iterrows():
        try:
            FeatureSchema(**row)
        except ValidationError as e:
            logging.error(f"Schema validation error: {e}")
            raise
    return df

def prepare_data(df, model_type, category=None, neighborhood=None):
    if category:
        df = df[df["service_category"] == category]
    if neighborhood:
        df = df[df["neighborhood"] == neighborhood]
    assert len(df) > 100, "Not enough data for training."
    if model_type == "prophet":
        df = df.copy()
        df["ds"] = pd.to_datetime(df["year"].astype(str) + "-" + df["week"].astype(str) + "-1", format="%Y-%W-%w")
        df["y"] = df["volume"]
        return df[["ds", "y", "rain"]]
    elif model_type == "xgboost":
        features = ["moving_avg_4w", "rain", "dayofweek"]
        X = df[features]
        y = df["volume"]
        return X, y
    else:
        raise ValueError("Unknown model_type")

def train_prophet(data):
    best_score = float("inf")
    best_model = None
    for cps in [0.01, 0.1, 0.5]:
        model = Prophet(changepoint_prior_scale=cps, weekly_seasonality=True)
        model.add_regressor("rain")
        try:
            model.fit(data)
            # Simple CV: last 20% as validation
            split = int(len(data) * 0.8)
            train, val = data.iloc[:split], data.iloc[split:]
            forecast = model.predict(val)
            rmse = mean_squared_error(val["y"], forecast["yhat"], squared=False)
            if rmse < best_score:
                best_score = rmse
                best_model = model
        except Exception as e:
            logging.warning(f"Prophet training failed for cps={cps}: {e}")
    return best_model

def train_xgboost(data):
    X, y = data
    param_grid = {
        "n_estimators": [50, 100],
        "learning_rate": [0.01, 0.1]
    }
    tscv = TimeSeriesSplit(n_splits=4)
    model = XGBRegressor(objective="reg:squarederror", random_state=42)
    grid = GridSearchCV(model, param_grid, cv=tscv, scoring="neg_root_mean_squared_error")
    grid.fit(X, y)
    return grid.best_estimator_

def hierarchical_forecast(df, model_type):
    models = {}
    # Global (city)
    if model_type == "prophet":
        data = prepare_data(df, "prophet")
        models["global"] = train_prophet(data)
    else:
        data = prepare_data(df, "xgboost")
        models["global"] = train_xgboost(data)
    # By neighborhood
    for n in df["neighborhood"].unique():
        if model_type == "prophet":
            data = prepare_data(df, "prophet", neighborhood=n)
            models[f"neigh_{n}"] = train_prophet(data)
        else:
            data = prepare_data(df, "xgboost", neighborhood=n)
            models[f"neigh_{n}"] = train_xgboost(data)
    # By category
    for c in df["service_category"].unique():
        if model_type == "prophet":
            data = prepare_data(df, "prophet", category=c)
            models[f"cat_{c}"] = train_prophet(data)
        else:
            data = prepare_data(df, "xgboost", category=c)
            models[f"cat_{c}"] = train_xgboost(data)
    # By category+neighborhood
    for c in df["service_category"].unique():
        for n in df["neighborhood"].unique():
            if model_type == "prophet":
                data = prepare_data(df, "prophet", category=c, neighborhood=n)
                models[f"{c}_{n}"] = train_prophet(data)
            else:
                data = prepare_data(df, "xgboost", category=c, neighborhood=n)
                models[f"{c}_{n}"] = train_xgboost(data)
    return models

def backtest(model, data, model_type):
    if model_type == "prophet":
        df = data.copy()
        tscv = TimeSeriesSplit(n_splits=4)
        metrics = []
        for train_idx, test_idx in tscv.split(df):
            train, test = df.iloc[train_idx], df.iloc[test_idx]
            m = Prophet(weekly_seasonality=True)
            m.add_regressor("rain")
            m.fit(train)
            forecast = m.predict(test)
            rmse = mean_squared_error(test["y"], forecast["yhat"], squared=False)
            mape = mean_absolute_percentage_error(test["y"], forecast["yhat"])
            lower = forecast["yhat_lower"]
            upper = forecast["yhat_upper"]
            coverage = np.mean((test["y"] >= lower) & (test["y"] <= upper))
            metrics.append((rmse, mape, coverage))
        avg = np.mean(metrics, axis=0)
        return {"rmse": avg[0], "mape": avg[1], "coverage": avg[2]}
    else:
        X, y = data
        tscv = TimeSeriesSplit(n_splits=4)
        metrics = []
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            m = XGBRegressor(objective="reg:squarederror", random_state=42)
            m.fit(X_train, y_train)
            y_pred = m.predict(X_test)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            mape = mean_absolute_percentage_error(y_test, y_pred)
            # CI coverage not available for XGBoost, set to np.nan
            metrics.append((rmse, mape, np.nan))
        avg = np.mean(metrics, axis=0)
        return {"rmse": avg[0], "mape": avg[1], "coverage": avg[2]}

def save_model(model, output_path):
    joblib.dump(model, output_path)
    logging.info(f"Model saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Treinamento de modelos de forecast Vancouver 311")
    parser.add_argument("--input", required=True, help="Path to features parquet file")
    parser.add_argument("--output", required=True, help="Path to save trained model (.pkl)")
    parser.add_argument("--model", required=True, choices=["prophet", "xgboost"], help="Model type")
    args = parser.parse_args()

    logging.info(f"Loading features from {args.input}")
    df = load_features(args.input)
    logging.info(f"Loaded {len(df)} rows")

    logging.info(f"Starting hierarchical forecast training with model: {args.model}")
    models = hierarchical_forecast(df, args.model)

    # Backtest global model
    if args.model == "prophet":
        data = prepare_data(df, "prophet")
    else:
        data = prepare_data(df, "xgboost")
    metrics = backtest(models["global"], data, args.model)
    logging.info(f"Backtest global model: {metrics}")

    save_model(models, args.output)
    print(f"Modelos treinados salvos em {args.output}")

if __name__ == "__main__":
    main()
