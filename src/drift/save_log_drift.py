import matplotlib.pyplot as plt
from src.utils.storage import path_validate, save_pickle
import seaborn as sns
import mlflow
import json
import numpy as np
from scipy.stats import entropy 


def compute_kl(pre_dict, post_dict):
    kl_dict = {}
    for feature in pre_dict.keys():
        categories = list(pre_dict[feature].keys())
        pre_values = np.array(list(pre_dict[feature].values())) + 1e-12  # avoid log(0)
        post_values = np.array([post_dict[feature].get(cat, 0) for cat in categories]) + 1e-12
        kl_dict[feature] = float(entropy(pre_values, post_values))  # KL(P||Q)
    return kl_dict


def plot_overlap(pre_dict, post_dict, ylabel, kind, plot_dir, kl_dict=None):
    for feature in pre_dict.keys():
        plt.figure(figsize=(10,5))
        categories = list(pre_dict[feature].keys())
        pre_values = list(pre_dict[feature].values())
        post_values = [post_dict[feature].get(cat, 0) for cat in categories]  # align categories

        width = 0.4
        x = range(len(categories))
        plt.bar([i - width/2 for i in x], pre_values, width=width, label="Reference", alpha=0.7)
        plt.bar([i + width/2 for i in x], post_values, width=width, label="Current", alpha=0.7)

        plt.xticks(x, categories, rotation=45)
        plt.xlabel(feature)
        plt.ylabel(ylabel)
        title = f"{feature} - Reference vs Current ({kind})"
        if kl_dict and feature in kl_dict:
            title += f" | KL={kl_dict[feature]:.4f}"
        plt.title(title)
        plt.legend()
        plt.tight_layout()

        plot_path = f"{plot_dir}/{feature}_{kind}_drift_overlap.png"
        plt.savefig(plot_path)
        plt.close()

def save_and_log_drift(
    feature_drift_pre, feature_drift_post,
    target_drift_pre, target_drift_post,
    base_path="drift_results", run_name="simulate_drift_run"
):
    # Ensure base directory exists
    path_validate(base_path)
    plot_dir = f"{base_path}/plots"
    path_validate(plot_dir)

    # --- 1. Save dictionaries ---
    save_pickle(feature_drift_pre, f"{base_path}/feature_drift_pre.pkl")
    save_pickle(feature_drift_post, f"{base_path}/feature_drift_post.pkl")
    save_pickle(target_drift_pre, f"{base_path}/target_drift_pre.pkl")
    save_pickle(target_drift_post, f"{base_path}/target_drift_post.pkl")

    for name, d in zip(
        ["feature_drift_pre", "feature_drift_post", "target_drift_pre", "target_drift_post"],
        [feature_drift_pre, feature_drift_post, target_drift_pre, target_drift_post]
    ):
        with open(f"{base_path}/{name}.json", "w") as f:
            json.dump(d, f, indent=4)

    # --- 2. Compute KL divergence ---
    kl_feature = {}
    kl_target = {}

    kl_feature = compute_kl(feature_drift_pre, feature_drift_post)
    kl_target = compute_kl(target_drift_pre, target_drift_post)

    # Save KL metrics
    with open(f"{base_path}/kl_feature.json", "w") as f:
        json.dump(kl_feature, f, indent=4)
    with open(f"{base_path}/kl_target.json", "w") as f:
        json.dump(kl_target, f, indent=4)

    # --- 3. Plot overlapped distributions ---
    plot_overlap(feature_drift_pre, feature_drift_post, ylabel="Probability", kind="feature", plot_dir=plot_dir, kl_dict=kl_feature)
    plot_overlap(target_drift_pre, target_drift_post, ylabel="Target Probability", kind="target", plot_dir=plot_dir, kl_dict=kl_target)

    # --- 4. Log everything to MLflow ---
    with mlflow.start_run(run_name=run_name):
        # Pickles
        for f in ["feature_drift_pre.pkl","feature_drift_post.pkl","target_drift_pre.pkl","target_drift_post.pkl"]:
            mlflow.log_artifact(f"{base_path}/{f}")
        # JSONs
        for f in ["feature_drift_pre.json","feature_drift_post.json","target_drift_pre.json","target_drift_post.json","kl_feature.json","kl_target.json"]:
            mlflow.log_artifact(f"{base_path}/{f}")
        # Plots
        for feature in feature_drift_pre.keys():
            mlflow.log_artifact(f"{plot_dir}/{feature}_feature_drift_overlap.png")
            mlflow.log_artifact(f"{plot_dir}/{feature}_target_drift_overlap.png")

    return kl_feature, kl_target
