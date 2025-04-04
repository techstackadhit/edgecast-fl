import subprocess
import time

# 🚀 Start the federated server
print("🚀 Launching Federated Server...")
server_proc = subprocess.Popen(["python", "fl/server.py"])

# Wait a few seconds to ensure the server is up
time.sleep(10)

# Launch 3 clients using different datasets (client_1.csv, client_2.csv, client_3.csv)
client_ids = ["client_1", "client_2", "client_3"]
client_procs = []

for client_id in client_ids:
    print(f"🟢 Launching Client {client_id}")
    proc = subprocess.Popen(["python", "fl/client_template.py", client_id])
    client_procs.append(proc)
    time.sleep(10)  # Add delay to avoid overlapping initialization

# Wait for all clients to finish training
for proc in client_procs:
    proc.wait()

# Terminate the server after all clients have completed
server_proc.terminate()
time.sleep(5)
print("✅ Federated Learning Simulation Complete.")
