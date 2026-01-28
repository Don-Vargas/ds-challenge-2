import matplotlib
matplotlib.use("Agg")

from src.model_experiments import model_metrics

def run_model_metrics(version='v1', dataset_name = 'ds4'):
    results = {
        "metrics": model_metrics.test_metrics(version, dataset_name),
        "model_keys": model_metrics.model_keys(version, dataset_name),
    }
    return results

def run_all_model_metrics(version='v1'):
    results = {
        "metrics": model_metrics.all_test_metrics(version),
        "model_keys": model_metrics.all_model_keys(version),
    }
    return results
