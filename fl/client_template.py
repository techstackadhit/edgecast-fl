import os
import csv
import flwr as fl
import pandas as pd
import sys
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt


# --- Load dataset for the given client ---
client_id = sys.argv[1]  # Example: "client_1"
df = pd.read_csv(f"./clients/{client_id}.csv", parse_dates=["date"])

# --- Feature engineering ---
features = ["humidity", "rainfall", "pressure"]
df["target_temp"] = df["temperature"].shift(-1)
df.dropna(inplace=True)

X = df[features]
y = df["target_temp"]

# --- Define Flower FL client class using MLPRegressor ---
class FLClient(fl.client.NumPyClient):
    def __init__(self):
        # Create MLPRegressor with warm_start so it continues training
        self.model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1, warm_start=True, random_state=42)

        # Dummy fit to initialize internal weights (so coefs_ & intercepts_ exist)
        self.model.fit(X[:5], y[:5])

    def get_parameters(self, config):
        # Return all weights and biases as a list
        return self.model.coefs_ + self.model.intercepts_

    def set_parameters(self, parameters):
        # Separate list into coefs_ and intercepts_ based on layer count
        n_layers = len(self.model.coefs_)
        self.model.coefs_ = parameters[:n_layers]
        self.model.intercepts_ = parameters[n_layers:]

    def fit(self, parameters, config):
        # Receive global model parameters and fit locally
        self.set_parameters(parameters)
        self.model.fit(X, y)
        return self.get_parameters(config), len(X), {}

    def evaluate(self, parameters, config):
        # Evaluate local model after aggregation
        self.set_parameters(parameters)
        predictions = self.model.predict(X)
        mse = mean_squared_error(y, predictions)
        rmse = sqrt(mse)  # Hitung manual RMSE
        print(f"📊 [{client_id}] RMSE: {rmse:.2f}")

         # Log to terminal
        print(f"📊 [{client_id}] RMSE: {rmse:.2f}")

        # Log to CSV 
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{client_id}_fedlog.csv")

        # Get current round from config if available
        current_round = config.get("rnd", "unknown")

        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(["round", "rmse"])
            writer.writerow([current_round, rmse])
        
        # Save predictions per client
        pred_df = pd.DataFrame({
            "actual_temp": y,
            "predicted_temp": predictions
        })

        pred_df.to_csv(f"logs/{client_id}_predictions.csv", index=False)


        return rmse, len(X), {"rmse": rmse}

# --- Start the Flower client ---
if __name__ == "__main__":
    print(f"🟢 Starting client: {client_id}")
    fl.client.start_numpy_client(server_address="localhost:8080", client=FLClient())
