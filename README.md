# EdgeCast-FL: Federated Learning for Local Weather Forecasting

**EdgeCast-FL** explores the implementation of **Federated Learning (FL)** for distributed weather prediction, simulating an edge computing environment using localized sensor data. Each client in the system represents a virtual edge node, collaboratively training a global model without sharing raw data. Emulating real-world edge-based deployment scenarios in a controlled simulation setting.

---

## Objectives

- Implement a Federated Learning pipeline using the [Flower](https://flower.dev/) framework.
- Predict daily temperature using local weather data from Jakarta (2023–2024) (source:https://www.visualcrossing.com/weather-data/)
- Compare model performance in centralized vs federated settings.
- Analyze convergence behavior across IID and (future) non-IID data distributions.

---

## Technologies Used

- Python 3.11+
- [Flower](https://flower.dev/) for Federated Learning orchestration
- Scikit-learn (ML models: MLPRegressor, LinearRegression)
- Pandas, NumPy, Matplotlib
- Jupyter Notebooks (for post-analysis)
- VS Code on Windows (local development)

---

## Project Structure

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

## Models Used

This project focuses on **regression-based temperature prediction** using the following models:

### Centralized Baseline
- **Model**: Multi-layer Perceptron Regressor (MLPRegressor)
- **Input**: Daily weather features (e.g., date, temp_min, humidity, etc.)
- **Output**: Predicted temperature (next day)

### Federated Learning Model
- **Same model architecture as baseline**
- Trained in a decentralized manner using:
  - **Federated Averaging (FedAvg)** strategy
  - Flower framework for simulation

### Model Notes
- MLP is selected based on initial performance testing (outperformed Linear Regression)
- Configurable hidden layers, activation functions, and learning rates
- Model evaluation is tracked using **RMSE** across multiple rounds

---

## Setup & Installation

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

## How to Run

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

## Analyze Results

After training, evaluation results and predictions are stored per client in `.csv` files inside the `logs/` directory.

You can perform:

- **Trend analysis**: RMSE per round across all clients
- **Ideal round detection**: using slope or moving average heuristics
- **Prediction evaluation**: compare predicted vs actual temperature per client
- **Error metrics**: MAE and RMSE comparison across clients
- **Visual analysis**: line charts and bar plots of model performance

### Tools Used
- **Notebooks**:
  - `notebooks/analyze_federated_logs.ipynb` – convergence & round trends
  - `notebooks/analyze_federated_predictions.ipynb` – prediction performance
- **Libraries**: Pandas, Matplotlib, Scikit-learn

The notebooks generate:
- RMSE curves over training rounds
- MAE/RMSE bar charts per client
- Actual vs Predicted temperature plots
- Summary tables and insight notes for further analysis

---

## Future Work / To-do

Here are several planned extensions and experiments:

- **Non-IID data simulation**  
  Clients will be assigned seasonal data (e.g. dry vs rainy periods)

- **Strategy comparison**
  - FedAvg (current)
  - FedProx (to be added)
  - Other Flower-compatible strategies

- **Centralized vs Federated benchmarking**
  - Accuracy, convergence, stability

- **Web dashboard integration**
  - Show RMSE trends
  - Notify on training completion

- **Model variations**
  - Test Random Forest, XGBoost, LSTM (future)

---

## Author / Credits

- **Author**: Aditya Arya Putranda
- **Year**: 2025  
- **Affiliation**: Independent Research 

## License

This project is licensed under the **MIT License** — you are free to use, modify, and distribute it with attribution.

See the [`LICENSE`](LICENSE) file for more details.
