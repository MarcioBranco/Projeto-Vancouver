#!/usr/bin/env python
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
import csv

ROOT = Path.home() / "proj-van"
DATA_RAW = ROOT / "data_raw"
DATA_RAW.mkdir(parents=True, exist_ok=True)

def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

def write_sample(path: Path) -> None:
    rows = [
        {"service_request_id": 1, "service_name": "Bulky Item Pickup", "status": "closed",
         "requested_datetime": "2025-01-03T10:15:00Z", "updated_datetime": "2025-01-04T12:20:00Z",
         "address":"Main St", "neighborhood":"Downtown", "latitude":49.2827, "longitude":-123.1207},
        {"service_request_id": 2, "service_name": "Graffiti Removal", "status": "open",
         "requested_datetime": "2025-01-05T09:02:00Z", "updated_datetime": "",
         "address":"Broadway", "neighborhood":"Kitsilano", "latitude":49.2680, "longitude":-123.1553},
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch 311 service requests CSV into ~/proj-van/data_raw/")
    parser.add_argument("--city", required=True, choices=["vancouver"])
    parser.add_argument("--source-url", default="")
    parser.add_argument("--outfile", default="")
    args = parser.parse_args()

    ts = _timestamp()
    outfile = args.outfile or f"{args.city}_311_{ts}.csv"
    out_path = DATA_RAW / outfile

    if args.source_url:
        df = pd.read_csv(args.source_url)
        df.to_csv(out_path, index=False)
    else:
        write_sample(out_path)

    print(str(out_path))

if __name__ == "__main__":
    main()
