import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("data/preprocessed/weather_jakarta_clean.csv")
df["target_temp"] = df["temperature"].shift(-1)
df.dropna(inplace=True)

X = df[["humidity", "rainfall", "pressure"]]
y = df["target_temp"]

# Split data
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Load best model
model = joblib.load("model/baseline_model.pkl")
y_pred = model.predict(X_test)

# Save actual vs predicted
pd.DataFrame({
    "actual_temp": y_test.values,
    "predicted_temp": y_pred
}).to_csv("logs/centralized_predictions.csv", index=False)