#!/usr/bin/env python
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path.home() / "proj-van"
DATA_RAW = ROOT / "data_raw"
DATA_RAW.mkdir(parents=True, exist_ok=True)

def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

def write_sample(path: Path) -> None:
    df = pd.DataFrame([
        {"fiscal_year": 2024, "department": "Sanitation", "program": "Bulky Item", "budget": 2500000},
        {"fiscal_year": 2024, "department": "Parks & Rec", "program": "Graffiti", "budget": 1200000},
    ])
    df.to_csv(path, index=False)

def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch municipal budget CSV into ~/proj-van/data_raw/")
    parser.add_argument("--city", required=True, choices=["vancouver"])
    parser.add_argument("--source-url", default="")
    parser.add_argument("--outfile", default="")
    args = parser.parse_args()

    ts = _timestamp()
    outfile = args.outfile or f"{args.city}_budget_{ts}.csv"
    out_path = DATA_RAW / outfile

    if args.source_url:
        df = pd.read_csv(args.source_url)
        df.to_csv(out_path, index=False)
    else:
        write_sample(out_path)

    print(str(out_path))

if __name__ == "__main__":
    main()
