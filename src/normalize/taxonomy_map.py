#!/usr/bin/env python
import os
import logging
import argparse
from pathlib import Path
import pandas as pd
from pydantic import BaseModel, ValidationError, field_validator

# Diretórios
ROOT = Path.home() / "proj-van"
DATA_RAW = ROOT / "data_raw"
DATA_PROCESSED = ROOT / "data_processed" / "normalized"
LOG_DIR = ROOT / "artifacts" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

# Logging
logging.basicConfig(
    filename=LOG_DIR / "normalize.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

# Dicionário de mapeamento de categorias
CATEGORY_MAP = {
    "Graffiti": "Urban Maintenance",
    "Pothole": "Road Repair",
    "Street Cleaning": "Urban Maintenance",
    "Bulky Item": "Waste Management",
    "Large Item": "Waste Management",
    "Mattress": "Waste Management",
    "Streetlight": "Urban Maintenance",
}

# Pydantic model para validação
class RequestRecord(BaseModel):
    service_type: str
    neighbourhood: str
    opened_date: str

    @field_validator('service_type', 'neighbourhood', 'opened_date')
    @classmethod
    def not_empty(cls, v):
        if not v or v == 'Unknown':
            raise ValueError('Field cannot be empty or Unknown')
        return v

def load_data(input_path):
    logging.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path, sep=';', on_bad_lines='skip')
    logging.info(f"Loaded {len(df)} rows from {input_path}")
    return df

def normalize_columns(df):
    """
    Renames and cleans columns to a standard format.
    """
    # Create a copy of the DataFrame to avoid the SettingWithCopyWarning
    df = df.copy()
    
    # Strip whitespace and convert column names to lowercase for robust matching
    df.columns = df.columns.str.strip().str.lower()
    
    # Define a robust mapping from common names to our standard names
    col_map = {
        'service request type': 'service_type',
        'neighbourhood': 'neighbourhood',
        'neighborhood': 'neighbourhood',
        'local area': 'neighbourhood',
        'opened': 'opened_date',
        'service request open timestamp': 'opened_date',
        'closed': 'closed_date',
        'service request close timestamp': 'closed_date',
        'address': 'address',
        'latitude': 'latitude',
        'longitude': 'longitude',
    }

    # Rename columns based on the robust map
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Fill null values for specified columns
    for col in ['service_type', 'neighbourhood', 'opened_date', 'closed_date', 'address', 'latitude', 'longitude']:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
    return df

def map_categories(df):
    def map_cat(val):
        for key, cat in CATEGORY_MAP.items():
            if pd.isna(val):
                return 'Other'
            if key.lower() in str(val).lower():
                return cat
        return 'Other'
    df['category'] = df['service_type'].apply(map_cat)
    mapped = df['category'].value_counts().to_dict()
    logging.info(f"Mapped categories: {mapped}")
    return df

def validate_data(df):
    errors = []
    for idx, row in df.iterrows():
        try:
            RequestRecord(
                service_type=row.get('service_type', 'Unknown'),
                neighbourhood=row.get('neighbourhood', 'Unknown'),
                opened_date=row.get('opened_date', 'Unknown')
            )
        except ValidationError as e:
            errors.append((idx, str(e)))
    if errors:
        for idx, err in errors:
            logging.error(f"Validation error at row {idx}: {err}")
        raise ValueError(f"Validation failed for {len(errors)} rows.")
    logging.info("Validation passed for all rows.")

def save_normalized(df, output_path):
    df.to_parquet(output_path, index=False)
    logging.info(f"Saved normalized data to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Normalize Vancouver 311 data")
    parser.add_argument('--input', required=True, help="Input CSV file or directory")
    parser.add_argument('--output', required=True, help="Output Parquet file or directory")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if input_path.is_dir():
        csv_files = list(input_path.glob("*.csv"))
        if not csv_files:
            logging.error(f"No CSV files found in {input_path}")
            return
        for csv_file in csv_files:
            df = load_data(csv_file)
            df = normalize_columns(df)
            df = map_categories(df)
            assert df['category'].notnull().all(), "Null values found in 'category'"
            validate_data(df)
            out_file = output_path / (csv_file.stem + "_normalized.parquet")
            save_normalized(df, out_file)
    else:
        df = load_data(input_path)
        df = normalize_columns(df)
        df = map_categories(df)
        assert df['category'].notnull().all(), "Null values found in 'category'"
        validate_data(df)
        if output_path.is_dir():
            out_file = output_path / (input_path.stem + "_normalized.parquet")
        else:
            out_file = output_path
        save_normalized(df, out_file)

if __name__ == "__main__":
    main()