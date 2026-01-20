import pandas as pd
from fastapi import FastAPI, BackgroundTasks
from src.drift.simulation.simulate_concept_drift import concept_drift_simulation
from api.schemas import TrainingRequest, DriftRequest, DriftResponse
from api.analyze import run_analysis
from api.model_metrics import run_model_metrics
from api.training_mode import start_training
from api.inference_mode import start_inference


app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello FastAPI"}


@app.post("/analyze")
def analyze():
    result = run_analysis()
    return {
        "status": "success",
        "result": result
    }


@app.post("/model-metrics")
def model_metrics_endpoint():
    return run_model_metrics()


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
def inference(background_tasks: BackgroundTasks):
    background_tasks.add_task(start_inference)
    return {"status": "inference started"}


@app.post("/simulate-drift", response_model=DriftResponse)
def simulate_drift(request: DriftRequest):

    reference_df, drifted_df, drift_results = concept_drift_simulation(
        file_path=request.file_path,
        features=request.features,
        target=request.target,
        alpha=request.alpha,
        return_data=True
    )

    return {
        "drift_results": drift_results.to_dict(orient="records"),
        "reference_data": reference_df.to_dict(orient="records"),
        "drifted_data": drifted_df.to_dict(orient="records"),
    }
