# driver_call_endpoint.py
import requests
from src.drift.save_log_drift import save_and_log_drift

# 1️ Define your endpoint URL
url = "http://127.0.0.1:8000/simulate-drift-file"  # Change to your actual API host

# 2️ Set query parameters
params = {
    "file_path": "current_inference/",
    "version": "v3",
    "selected_ds": "ds4"
}

# 3️ Call the endpoint
response = requests.post(url, params=params)
data = response.json()

# 4️ Check if the call was successful
if data.get("status") == "success":
    feature_drift = data["feature_drift"]
    target_drift = data["target_drift"]

    # 5️ Save and log to MLflow
    save_and_log_drift(
        feature_drift=feature_drift,
        target_drift=target_drift,
        base_path="src/drift/drift_results",
        run_name="simulate_drift_run"
    )

    print("Drift simulation saved and logged successfully.")

else:
    print("Error:", data.get("message"))
