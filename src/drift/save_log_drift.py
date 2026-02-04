import matplotlib.pyplot as plt
from src.utils.storage import path_validate, save_pickle
import seaborn as sns
import mlflow
import json

def save_and_log_drift(feature_drift, target_drift, base_path="drift_results", run_name="simulate_drift_run"):
    # Ensure base directory exists
    path_validate(base_path)

    # --- 1. Save dictionaries as pickle ---
    save_pickle(feature_drift, f"{base_path}/feature_drift.pkl")
    save_pickle(target_drift, f"{base_path}/target_drift.pkl")

    # Save dictionaries as JSON
    with open(f"{base_path}/feature_drift.json", "w") as f:
        json.dump(feature_drift, f, indent=4)
    with open(f"{base_path}/target_drift.json", "w") as f:
        json.dump(target_drift, f, indent=4)

    # --- 2. Create plots ---
    plot_dir = f"{base_path}/plots"
    path_validate(plot_dir)

    # Feature drift plots
    for feature, probs in feature_drift.items():
        plt.figure(figsize=(10, 5))
        categories = list(probs.keys())
        values = list(probs.values())
        sns.barplot(x=categories, y=values)
        plt.title(f"Feature Distribution Before Drift: {feature}")
        plt.xlabel(feature)
        plt.ylabel("Probability")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plot_path = f"{plot_dir}/{feature}_feature_drift.png"
        plt.savefig(plot_path)
        plt.close()

    # Target drift plots
    for feature, probs in target_drift.items():
        plt.figure(figsize=(10, 5))
        categories = list(probs.keys())
        values = list(probs.values())
        sns.barplot(x=categories, y=values)
        plt.title(f"Target Probability per Feature Category: {feature}")
        plt.xlabel(feature)
        plt.ylabel("Target Probability")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plot_path = f"{plot_dir}/{feature}_target_drift.png"
        plt.savefig(plot_path)
        plt.close()

    # --- 3. Log to MLflow ---
    with mlflow.start_run(run_name=run_name):
        # Log dictionaries as artifacts
        mlflow.log_artifact(f"{base_path}/feature_drift.pkl")
        mlflow.log_artifact(f"{base_path}/target_drift.pkl")
        mlflow.log_artifact(f"{base_path}/feature_drift.json")
        mlflow.log_artifact(f"{base_path}/target_drift.json")

        # Log all plots
        for feature in feature_drift.keys():
            mlflow.log_artifact(f"{plot_dir}/{feature}_feature_drift.png")
            mlflow.log_artifact(f"{plot_dir}/{feature}_target_drift.png")
