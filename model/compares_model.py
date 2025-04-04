import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from math import sqrt
import joblib
import os

# Load preprocessed dataset
df = pd.read_csv("./data/preprocessed/weather_jakarta_clean.csv", parse_dates=["date"])

# Use selected features
features = ["humidity", "rainfall", "pressure"]
df["target_temp"] = df["temperature"].shift(-1)
df.dropna(inplace=True)

X = df[features]
y = df["target_temp"]

# Split dataset into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Define models to test
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
    "MLPRegressor": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
}

# Track best model
best_model = None
best_model_name = ""
best_rmse = float("inf")

# Create output dir
os.makedirs("model", exist_ok=True)

# Train and evaluate each model
for name, model in models.items():
    print(f"\n🔍 Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"✅ {name} - MAE: {mae:.2f} | RMSE: {rmse:.2f}")
    
    # Save model
    joblib.dump(model, f"model/{name}.pkl")

    # Update best model
    if rmse < best_rmse:
        best_rmse = rmse
        best_model = model
        best_model_name = name

# Save best model as baseline
joblib.dump(best_model, "model/baseline_model.pkl")
print(f"\n🏆 Best model: {best_model_name} (RMSE: {best_rmse:.2f}) saved as 'baseline_model.pkl'")
