import flwr as fl
# Start the FL server

def get_eval_config(rnd: int):
    return {"rnd": rnd}

if __name__ == "__main__":
    print("🚀 Starting FL Server...")

    # Define simple FedAvg strategy
    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=3,
        min_available_clients=3,
        on_fit_config_fn=lambda rnd: {"rnd": rnd}, # send number of round to client
        on_evaluate_config_fn=get_eval_config
    )

    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=20, round_timeout=60),
        strategy=strategy
    )

    print("🛑 Server shut down.")
