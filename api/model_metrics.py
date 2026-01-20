import matplotlib
matplotlib.use("Agg")  # 👈 IMPORTANT

from src.model_experiments import model_metrics

def run_model_metrics():
    results = {
        "metrics": model_metrics.test_metrics(),
        "model_keys": model_metrics.model_keys(),
    }
    return results