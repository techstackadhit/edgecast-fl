# 🌤️ EdgeCast-FL: Federated Learning for Local Weather Forecasting

**EdgeCast-FL** explores the implementation of **Federated Learning (FL)** for distributed weather prediction using edge-based sensor data. The system simulates decentralized environments where each client represents a local node, collaboratively training a global model without sharing raw data.

---

## 🎯 Objectives

- Implement a Federated Learning pipeline using the [Flower](https://flower.dev/) framework.
- Predict daily temperature using local weather data from Jakarta (2023–2024) (source:https://www.visualcrossing.com/weather-data/)
- Compare model performance in centralized vs federated settings.
- Analyze convergence behavior across IID and (future) non-IID data distributions (soon).

---

## 🛠️ Technologies Used

- Python 3.11+
- [Flower](https://flower.dev/) for Federated Learning orchestration
- Scikit-learn (ML models: MLPRegressor, LinearRegression)
- Pandas, NumPy, Matplotlib
- Jupyter Notebooks (for post-analysis)
- VS Code on Windows (local development)

---

## 📁 Project Structure

```
edgecast-fl/
├── fl/                     # FL logic: server, client, simulation
├── models/                 # shared model configs
├── scripts/                # Data preprocessing and splitting
├── data/                   # Raw and cleaned datasets
│   ├── raw/
│   └── preprocessed/
├── clients/                # Client-specific datasets
├── logs/                   # RMSE logs per round
├── notebooks/              # Evaluation and plotting notebooks
├── requirements.txt        # Python dependencies
├── .gitignore              # venv, pycache, logs, etc
└── README.md               # You're reading it!
```

---

## 🧠 Models Used

This project focuses on **regression-based temperature prediction** using the following models:

### 🔹 Centralized Baseline
- **Model**: Multi-layer Perceptron Regressor (MLPRegressor)
- **Input**: Daily weather features (e.g., date, temp_min, humidity, etc.)
- **Output**: Predicted temperature (next day)

### 🔹 Federated Learning Model
- **Same model architecture as baseline**
- Trained in a decentralized manner using:
  - **Federated Averaging (FedAvg)** strategy
  - Flower framework for simulation

### 🔧 Model Notes
- MLP is selected based on initial performance testing (outperformed Linear Regression)
- Configurable hidden layers, activation functions, and learning rates
- Model evaluation is tracked using **RMSE** across multiple rounds

---

## ⚙️ Setup & Installation

It's recommended to use a virtual environment:

```
# Create and activate virtual environment
python -m venv venv 
venv\Scripts\activate # Windows: 
source venv/bin/activate # macOS/Linux:

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 How to Run

### 1. Start the Federated Server
```
python fl/server.py
```

### 2. Launch Simulated Clients
```
python fl/run_simulation.py
```

### 3. Analyze Training Results
Open the analysis notebook:
```
notebooks/analyze_federated_logs.ipynb
```

---

## 📊 Analyze Results

After training, results are logged per client in `.csv` files (one per client) inside the `logs/` directory.

You can perform:

- 📈 **Trend analysis**: RMSE per round
- 🎯 **Ideal round detection**: using slope or moving average
- 📊 **Client comparison**: consistency, convergence, and divergence across clients

### Tools Used
- **Notebook**: `notebooks/analyze_federated_logs.ipynb`
- **Libraries**: Pandas, Matplotlib

The notebook generates:
- RMSE curves
- Summary tables
- Insight texts (e.g., stable clients, round thresholds)

---

## 🔍 Future Work / To-do

Here are several planned extensions and experiments:

- 🔄 **Non-IID data simulation**  
  Clients will be assigned seasonal data (e.g. dry vs rainy periods)

- 🔁 **Strategy comparison**
  - FedAvg (current)
  - FedProx (to be added)
  - Other Flower-compatible strategies

- 📊 **Centralized vs Federated benchmarking**
  - Accuracy, convergence, stability

- 🌐 **Web dashboard integration**
  - Show RMSE trends
  - Notify on training completion

- ⚡ **Model variations**
  - Test Random Forest, XGBoost, LSTM (future)

---

## 👤 Author / Credits

- **Author**: Aditya Arya Putranda
- **Year**: 2025  
- **Affiliation**: Independent Research 

## 📄 License

This project is licensed under the **MIT License** — you are free to use, modify, and distribute it with attribution.

See the [`LICENSE`](LICENSE) file for more details.