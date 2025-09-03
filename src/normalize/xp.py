#!/usr/bin/env python
import pandas as pd
from pathlib import Path

ROOT = Path.home() / "proj-van"
DATA_RAW = ROOT / "data_raw"

for csv_file in DATA_RAW.glob("3-1-1-service-requests*.csv"):
    try:
        df = pd.read_csv(csv_file, nrows=5, on_bad_lines='skip')
        print(f"\nArquivo: {csv_file.name}")
        print("Colunas:", df.columns.tolist())
    except Exception as e:
        print(f"Erro lendo {csv_file.name}: {e}")
