import numpy as np
from fastapi import FastAPI, BackgroundTasks, Query
from api.schemas import TrainingRequest
from api.analyze import run_analysis
from api.model_metrics import run_model_metrics, run_all_model_metrics
from api.training_mode import start_training
from api.inference_mode import start_inference
from src.drift.concept_drift import simulate_drift_auto
from fastapi.responses import JSONResponse



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
    version: str = Query("v1", description="Model version to use"),
    selected_ds: str = Query("ds4", description="Dataset to use for inference"),
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="Decision threshold"),
    background_tasks: BackgroundTasks = None,
):
    background_tasks.add_task(
        start_inference,
        version,
        selected_ds,
        threshold,
    )

    return {
        "status": "inference started",
        "version": version,
        "selected_ds": selected_ds,
        "threshold": threshold,
    }

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
def simulate_drift_file(
    file_path: str = Query("current_inference/", description="Path for inference data"),
    version: str = Query("v1", description="Model version"),
    selected_ds: str = Query("ds4", description="Dataset"),
):
    try:
        # Run the drift simulation
        original_df, drifted_df, feature_drift_dict, target_drift_dict = simulate_drift_auto(
            df_path=file_path,
            version=version,
            selected_ds=selected_ds,
        )

        # Optionally, you can convert dataframes to JSON if you want to return them
        return {
            "status": "success",
            "original_data_shape": original_df.shape,
            "drifted_data_shape": drifted_df.shape,
            "feature_drift": feature_drift_dict,
            "target_drift": target_drift_dict,
        }

    except FileNotFoundError:
        return JSONResponse(status_code=404, content={"status": "error", "message": f"File not found: {file_path}"})

    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
