#!/usr/bin/env python
import subprocess
import logging
import argparse
import time
import os
from pathlib import Path

ROOT = Path.home() / "proj-van"
LOG_DIR = ROOT / "artifacts/logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "pipeline.log"

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def run_step(script: str, args: list[str]):
    cmd = ['python', str(script)] + args
    for attempt in range(1, 4):
        try:
            logging.info(f"Running: {' '.join(cmd)} (Attempt {attempt})")
            result = subprocess.run(cmd, check=True)
            logging.info(f"Step succeeded: {' '.join(cmd)}")
            return
        except subprocess.CalledProcessError as e:
            logging.error(f"Step failed (attempt {attempt}): {e}")
            if attempt < 3:
                time.sleep(5)
            else:
                raise

def check_output(path: str):
    if not os.path.exists(path):
        logging.error(f"Expected output not found: {path}")
        raise FileNotFoundError(f"Output not found: {path}")
    logging.info(f"Output exists: {path}")

def run_pipeline():
    # 1. Normalizar CSV 311
    norm_in = str(ROOT / "data_raw/3-1-1-service-requests.csv")
    norm_out = str(ROOT / "data_processed/normalized/normalized_requests.parquet")
    run_step(ROOT / "src/normalize/taxonomy_map.py", ["--input", norm_in, "--output", norm_out])
    check_output(norm_out)

    # 2. Gerar features
    feat_in = norm_out
    feat_out = str(ROOT / "data_processed/features/features.parquet")
    run_step(ROOT / "src/features/build_features.py", ["--input", feat_in, "--output", feat_out])
    check_output(feat_out)

    # 3. Treinar modelo
    model_out = str(ROOT / "artifacts/models/forecast_v1.pkl")
    run_step(
        ROOT / "src/models/train_forecast.py",
        ["--input", feat_out, "--output", model_out, "--model", "prophet"]
    )
    check_output(model_out)

    # 4. Avaliação
    metrics_out = str(ROOT / "artifacts/metrics/eval_metrics.json")
    run_step(
        ROOT / "src/models/eval.py",
        ["--model", model_out, "--features", feat_out, "--output", metrics_out]
    )
    check_output(metrics_out)

    # 5. Relatório
    pdf_out = str(ROOT / "artifacts/reports/report.pdf")
    social_out = str(ROOT / "artifacts/reports/carrossel")
    run_step(
        ROOT / "src/reports/render_report.py",
        [
            "--features", feat_out,
            "--metrics", metrics_out,
            "--model", model_out,
            "--output-pdf", pdf_out,
            "--output-social", social_out
        ]
    )
    check_output(pdf_out)
    if not os.path.exists(social_out):
        logging.warning(f"Social output folder not found: {social_out}")
    else:
        logging.info(f"Social output exists: {social_out}")

def main():
    parser = argparse.ArgumentParser(description="Pipeline Projeto Vancouver 311")
    parser.add_argument("--schedule", type=str, help="Horário para rodar (formato HH:MM, opcional)")
    args = parser.parse_args()

    if args.schedule:
        target = args.schedule.strip()
        logging.info(f"Agendamento solicitado para {target}")
        while True:
            now = time.strftime("%H:%M")
            if now == target:
                logging.info("Horário agendado atingido, iniciando pipeline.")
                break
            time.sleep(30)
    try:
        run_pipeline()
        logging.info("Pipeline finalizado com sucesso.")
    except Exception as e:
        logging.exception(f"Pipeline falhou: {e}")
        exit(1)

if __name__ == "__main__":
    main()
