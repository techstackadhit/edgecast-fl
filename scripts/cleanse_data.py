import pandas as pd
import os

# Path input/output
raw_path = "./data/raw/Jakarta 2023-01-01 to 2024-12-31.csv"
output_path = "./data/preprocessed/weather_jakarta_clean.csv"

# ensure preprocessed data folder exist
os.makedirs("./data/preprocessed", exist_ok=True)

# Load data
df = pd.read_csv(raw_path)

# extract imporatnt columns
df_clean = df[["datetime", "temp", "humidity", "precip", "sealevelpressure"]].copy()

# rename columns
df_clean.rename(columns={
    "datetime": "date",
    "temp": "temperature",
    "precip": "rainfall",
    "sealevelpressure": "pressure"
}, inplace=True)

# Parse date
df_clean['date'] = pd.to_datetime(df_clean['date'], errors='coerce')

# drop empty rows
df_clean.dropna(inplace=True)

# sort and reset index
df_clean = df_clean.sort_values('date').reset_index(drop=True)

# sace output
df_clean.to_csv(output_path, index=False)

print(f"✅ Cleansed data has been saved to : {output_path}")
