from src.utils.storage import load_pickle, path_validate
import matplotlib.pyplot as plt
import mlflow
from src.modeling.modeling import EXPERIMENT_NAME
from sklearn.metrics import (
    accuracy_score, precision_score, f1_score, roc_auc_score,
    roc_curve, confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, average_precision_score,
)


# -------------------------
# Helper functions
# -------------------------

def compute_classification_metrics(y_test, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall / TPR": tpr,
        "TNR / Specificity": tnr,
        "FPR": fpr,
        "FNR": fnr,
        "F1-score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
        "Average Precision": average_precision_score(y_test, y_proba),
    }
    return y_pred, metrics


def save_plot(fig, filepath):
    path_validate(filepath)
    fig.savefig(filepath, dpi=300)
    plt.close(fig)


def plot_confusion_matrix(y_test, y_pred, filepath, title="Confusion Matrix"):
    fig, ax = plt.subplots()
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    plt.title(title)
    plt.tight_layout()
    save_plot(fig, filepath)


def plot_roc_curve(y_test, y_proba, filepath, title="ROC Curve"):
    fpr_vals, tpr_vals, _ = roc_curve(y_test, y_proba)
    auc_val = roc_auc_score(y_test, y_proba)
    fig, ax = plt.subplots()
    ax.plot(fpr_vals, tpr_vals, label=f"AUC = {auc_val:.4f}")
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    plt.tight_layout()
    save_plot(fig, filepath)


def plot_precision_recall_curve(y_test, y_proba, filepath, title="Precision-Recall Curve"):
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
    ap_val = average_precision_score(y_test, y_proba)
    fig, ax = plt.subplots()
    ax.plot(recall_vals, precision_vals, label=f"AP = {ap_val:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="upper right")
    plt.tight_layout()
    save_plot(fig, filepath)


def load_experiment_results(version='v1'):
    """Load experiment results pickle and return output path."""
    experiment_results = load_pickle(f'src/modeling/{version}/model_experiment_results.pkl')
    output_path = f'training_parameter_results/{version}/metric_figures/'
    return experiment_results, output_path


# -------------------------
# Endpoint: model_keys
# -------------------------
def model_keys(version, dataset_name):
    experiment_results, output_path = load_experiment_results(version)

    print("Top-level keys:", experiment_results.keys())
    print(f"Keys for dataset {dataset_name}:", experiment_results[dataset_name].keys())
    print("all_model_results:", experiment_results[dataset_name].get('all_model_results', "'all_model_results' key not found"))

    return f"results on: {output_path}"


# -------------------------
# Endpoint: test_metrics
# -------------------------
def test_metrics(version, dataset_name):
    experiment_results, output_path = load_experiment_results(version)

    y_test, y_test_proba = experiment_results[dataset_name]['y_test_proba']
    y_pred, metrics = compute_classification_metrics(y_test, y_test_proba)

    print("Classification Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # Plots
    plot_confusion_matrix(y_test, y_pred, f"{output_path}{dataset_name}/confusion_matrix.png")
    plot_roc_curve(y_test, y_test_proba, f"{output_path}{dataset_name}/roc_curve.png")
    plot_precision_recall_curve(y_test, y_test_proba, f"{output_path}{dataset_name}/prec_rec_curve.png")

    return f"results on: {output_path}{dataset_name}/"


# -------------------------
# Endpoint: all_model_keys
# -------------------------
def all_model_keys(version='v1'):
    experiment_results, output_path = load_experiment_results(version)
    for dataset_name, dataset_results in experiment_results.items():
        print(f"\nDataset: {dataset_name}")
        print("Top-level keys:", dataset_results.keys())
        print("all_model_results:", dataset_results.get('all_model_results', "'all_model_results' key not found"))
    return f"results on: {output_path}"


# -------------------------
# Endpoint: all_test_metrics
# -------------------------
def all_test_metrics(version='v1'):
    experiment_results, output_base = load_experiment_results(version)
    mlflow.set_experiment(EXPERIMENT_NAME)

    for dataset_name, results in experiment_results.items():
        print(f"\n=== Processing dataset: {dataset_name} ===")

        y_test, y_proba = results['y_test_proba']
        y_pred, metrics = compute_classification_metrics(y_test, y_proba)

        # MLflow logging
        with mlflow.start_run(run_name=f"inference_{version}_{dataset_name}"):
            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            output_path = f"{output_base}{dataset_name}/"
            plot_confusion_matrix(y_test, y_pred, f"{output_path}confusion_matrix.png",
                                  title=f"Confusion Matrix - {dataset_name}")
            mlflow.log_artifact(f"{output_path}confusion_matrix.png")

            plot_roc_curve(y_test, y_proba, f"{output_path}roc_curve.png",
                           title=f"ROC Curve - {dataset_name}")
            mlflow.log_artifact(f"{output_path}roc_curve.png")

            plot_precision_recall_curve(y_test, y_proba, f"{output_path}prec_rec_curve.png",
                                        title=f"Precision-Recall Curve - {dataset_name}")
            mlflow.log_artifact(f"{output_path}prec_rec_curve.png")

        print(f"Metrics for dataset {dataset_name}:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

    return f"All dataset metrics processed and logged under MLflow experiment: {EXPERIMENT_NAME}"
