#!/usr/bin/env python
import argparse
import logging
from pathlib import Path
import json

import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from pydantic import BaseModel, ValidationError

ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)
(ARTIFACTS / "logs").mkdir(exist_ok=True)
(ARTIFACTS / "plots").mkdir(exist_ok=True)
(ARTIFACTS / "metrics").mkdir(exist_ok=True)

LOG_FILE = ARTIFACTS / "logs/eval.log"
PLOT_FILE = ARTIFACTS / "plots/eval_plot.png"
DEFAULT_METRICS_FILE = ARTIFACTS / "metrics/eval_metrics.json"

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

class Metrics(BaseModel):
    rmse: float
    mape: float
    coverage: float

def load_model(model_path):
    logging.info(f"Loading model from {model_path}")
    return joblib.load(model_path)

def load_features(input_path):
    logging.info(f"Loading features from {input_path}")
    return pd.read_parquet(input_path)

def evaluate(model, data):
    logging.info("Evaluating model")
    X = data.drop(columns=["y", "ds"])
    y_true = data["y"].values
    y_pred = model.predict(X)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    # 80% CI: assume model has predict_interval or fallback to +/- 1.28*std
    if hasattr(model, "predict_interval"):
        pred_int = model.predict_interval(X, alpha=0.2)
        lower = pred_int["lower"]
        upper = pred_int["upper"]
    else:
        std = y_pred.std() if hasattr(y_pred, "std") else pd.Series(y_pred).std()
        lower = y_pred - 1.28 * std
        upper = y_pred + 1.28 * std
    coverage = ((y_true >= lower) & (y_true <= upper)).mean()
    metrics = {"rmse": rmse, "mape": mape, "coverage": coverage}
    logging.info(f"Metrics: {metrics}")
    return metrics, y_true, y_pred

def benchmark(data):
    logging.info("Running benchmarks")
    df = data.copy()
    df = df.sort_values("ds")
    df["ds"] = pd.to_datetime(df["ds"])
    # Naive: média da última semana
    last_week = df[df["ds"] >= (df["ds"].max() - pd.Timedelta(days=7))]
    naive_pred = last_week["y"].mean()
    # Seasonal naive: média da mesma semana do ano anterior
    df["week"] = df["ds"].dt.isocalendar().week
    df["year"] = df["ds"].dt.year
    this_week = df["week"].max()
    last_year = df["year"].max() - 1
    seasonal_naive = df[(df["week"] == this_week) & (df["year"] == last_year)]["y"].mean()
    logging.info(f"Naive forecast: {naive_pred}, Seasonal naive: {seasonal_naive}")
    return naive_pred, seasonal_naive

def generate_plots(metrics, y_true, y_pred, output_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(y_true, label="Real")
    plt.plot(y_pred, label="Previsto")
    plt.title(f"Pred vs Real (RMSE={metrics['rmse']:.2f}, MAPE={metrics['mape']:.2%})")
    plt.legend()
    plt.tight_layout()
    plot_path = Path(output_dir) / "eval_plot.png"
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"Plot saved to {plot_path}")

def save_metrics(metrics, output_path):
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logging.info(f"Metrics saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Avaliação Projeto Vancouver 311")
    parser.add_argument("--model", required=True, help="Caminho do modelo .pkl")
    parser.add_argument("--features", required=True, help="Caminho features .parquet")
    parser.add_argument("--output", default=str(DEFAULT_METRICS_FILE), help="Caminho para salvar métricas .json")
    args = parser.parse_args()

    try:
        model = load_model(args.model)
        data = load_features(args.features)
        metrics, y_true, y_pred = evaluate(model, data)
        # Validação Pydantic
        validated = Metrics(**metrics)
        assert validated.mape < 1.0, "MAPE acima do limite"
        save_metrics(metrics, args.output)
        generate_plots(metrics, y_true, y_pred, ARTIFACTS / "plots")
        naive_pred, seasonal_naive = benchmark(data)
        logging.info(f"Benchmarks - Naive: {naive_pred}, Seasonal Naive: {seasonal_naive}")
        print(f"Métricas salvas em {args.output}")
    except (AssertionError, ValidationError) as e:
        logging.error(f"Erro de validação: {e}")
        print(f"Erro de validação: {e}")
    except Exception as e:
        logging.error(f"Erro inesperado: {e}")
        print(f"Erro inesperado: {e}")

if __name__ == "__main__":
    main()
