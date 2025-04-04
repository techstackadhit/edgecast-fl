import pandas as pd
import os

# Input path for the preprocessed dataset
input_path = "./data/preprocessed/weather_jakarta_clean.csv"

# Output directory for the split client datasets
output_dir = "./clients"
os.makedirs(output_dir, exist_ok=True)

# Load the preprocessed weather dataset
df = pd.read_csv(input_path, parse_dates=["date"])

# Ensure the data is sorted by date
df = df.sort_values("date")

# Define date ranges for each client simulation
client_1 = df[(df['date'] >= '2023-01-01') & (df['date'] <= '2023-06-30')]
client_2 = df[(df['date'] >= '2023-07-01') & (df['date'] <= '2023-12-31')]
client_3 = df[(df['date'] >= '2024-01-01') & (df['date'] <= '2024-12-31')]

# Save each split to separate CSV files for simulated edge nodes
client_1.to_csv(f"{output_dir}/client_1.csv", index=False)
client_2.to_csv(f"{output_dir}/client_2.csv", index=False)
client_3.to_csv(f"{output_dir}/client_3.csv", index=False)

print("✅ Dataset successfully split into 3 client files and saved to './clients/'")
