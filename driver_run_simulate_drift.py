# driver_call_endpoint.py
import requests
from src.drift.save_log_drift import save_and_log_drift

# 1️ Define your endpoint URL
url = "http://127.0.0.1:8000/simulate-drift-file"  

# 2️ Set query parameters
params = {
    "file_path": "current_inference/",
    "version": "v3",
    "selected_ds": "ds4"
}

resp = requests.post(url, params=params)
data = resp.json()

if data["status"] == "success":
    save_and_log_drift(
        feature_drift_pre=data["feature_drift_pre"],
        feature_drift_post=data["feature_drift_post"],
        target_drift_pre=data["target_drift_pre"],
        target_drift_post=data["target_drift_post"],
        base_path="src/drift/drift_results",
        experiment_name="drift_experiment"
    )
    print("Drift logged and plots generated!")
else:
    print("Error:", data.get("message"))
