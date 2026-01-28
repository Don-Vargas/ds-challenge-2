import pandas as pd
import numpy as np
from fastapi import FastAPI, BackgroundTasks, Query
from api.schemas import TrainingRequest
from api.analyze import run_analysis
from api.model_metrics import run_model_metrics, run_all_model_metrics
from api.training_mode import start_training
from api.inference_mode import start_inference
from src.drift.concept_drift import simulate_drift_auto


app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello FastAPI"}


@app.post("/analyze")
def analyze_endpoint(version: str = "v1"):
    result = run_analysis(version)
    return {
        "status": "success",
        "result": result
    }


@app.post("/best_model-metrics")
def model_metrics_endpoint(
    version: str = Query("v1", description="Model version to use"),
    dataset_name: str = Query("ds4", description="Dataset name"),
):
    return run_model_metrics(version=version, dataset_name=dataset_name)


@app.post("/all_model-metrics")
def all_model_metrics_endpoint(
    version: str = Query(..., description="Model version to use"),
    *,
    background_tasks: BackgroundTasks,
):
    background_tasks.add_task(run_all_model_metrics, version)
    return {
        "status": "all model metrics saved",
        "version": version,
    }


@app.post("/training")
def training(
    payload: TrainingRequest,
    background_tasks: BackgroundTasks,
    ):
    background_tasks.add_task(
        start_training,
        payload.version,
        "target",
        payload.test_size,
    )
    return {
        "status": "training started",
        "version": payload.version,
        "test_size": payload.test_size,
    }


@app.post("/inference")
def inference(
    version: str = Query(..., description="Model version to use"),
    *,
    background_tasks: BackgroundTasks,
):
    background_tasks.add_task(start_inference, version)
    return {"status": "inference started", "version": version}

def clean_dict(d):
    """
    Convert all nested dict values to Python floats and keys to strings.
    Replace NaNs and infinite values with 0.0
    """
    clean = {}
    for k, v in d.items():
        clean_k = str(k)
        clean_v = {}
        for kk, vv in v.items():
            try:
                val = float(vv)
                if np.isnan(val) or np.isinf(val):
                    val = 0.0
            except:
                val = 0.0
            clean_v[str(kk)] = val
        clean[clean_k] = clean_v
    return clean

@app.post("/simulate-drift-file")
def simulate_drift_file(file_path: str):
    try:
        df, df_drifted, feature_drift_dict, target_drift_dict = simulate_drift_auto(df_path=file_path)
        
        # Make numeric columns finite and fill NaN
        df = df.replace([np.inf, -np.inf], 0).fillna(0)
        df_drifted = df_drifted.replace([np.inf, -np.inf], 0).fillna(0)
        
        feature_drift_dict = clean_dict(feature_drift_dict)
        target_drift_dict = clean_dict(target_drift_dict)
        
        response = {
            "status": "success",
            "df": df.to_dict(orient='records'),
            "drifted_data": df_drifted.to_dict(orient='records'),
            "feature_drift": feature_drift_dict,
            "target_drift": target_drift_dict
        }
        return response
        
    except FileNotFoundError:
        return {"status": "error", "message": f"File not found: {file_path}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
