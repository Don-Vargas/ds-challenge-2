from src.utils.storage import load_pickle, path_validate
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, average_precision_score,
)


def model_keys():
    experiment_results = load_pickle('src/modeling/v1/model_experiment_results.pkl')
    output_path ='training_parameter_results/v1/metric_figures/'
    dataset_name = 'ds4'

    print(experiment_results[dataset_name].keys())
    print(experiment_results[dataset_name]['all_model_results'])




def test_metrics():
    experiment_results = load_pickle('src/modeling/v1/model_experiment_results.pkl')
    output_path ='training_parameter_results/v1/metric_figures/'
    dataset_name = 'ds4'

    y_test, y_test_proba = experiment_results[dataset_name]['y_test_proba']

    # Convert probabilities to predicted classes (default threshold 0.5)
    y_pred = (y_test_proba >= 0.5).astype(int)

    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # Compute rates
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate (Recall)
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate (Specificity)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate

    # Classification metrics
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall / TPR": tpr,
        "TNR / Specificity": tnr,
        "FPR": fpr,
        "FNR": fnr,
        "F1-score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_test_proba),
    }

    # Print metrics
    print("Classification Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.tight_layout()

    file_path = f"{output_path}{dataset_name}/confusion_matrix.png"
    path_validate(file_path)

    plt.savefig(file_path, dpi=300)
    plt.close()

    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {metrics['ROC-AUC']:.4f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")

    plt.tight_layout()

    file_path = f"{output_path}{dataset_name}/roc_curve.png"
    path_validate(file_path)

    plt.savefig(file_path, dpi=300)
    plt.close()

    # Precision-Recall curve
    precision, recall, pr_thresholds = precision_recall_curve(y_test, y_test_proba)
    avg_precision = average_precision_score(y_test, y_test_proba)
    plt.figure()
    plt.plot(recall, precision, label=f"AP = {avg_precision:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="upper right")
    plt.tight_layout()

    file_path = f"{output_path}{dataset_name}/prec_rec_curve.png"
    path_validate(file_path)

    plt.savefig(file_path, dpi=300)
    plt.close()
