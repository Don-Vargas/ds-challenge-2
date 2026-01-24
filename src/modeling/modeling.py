import logging
from typing import Dict, List, Optional

import mlflow
import mlflow.sklearn
import pandas as pd

from src.model_experiments import experiments
from src.utils.storage import (
    ingest_data,
    save_pickle,
    load_pickle,
    export_data,
)


# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
EXPERIMENT_NAME = "binary_classification_modeling"

DATASETS: List[str] = [
    "ds1", "ds2", "ds3", "ds4", "ds5",
    "ds6", "ds7", "ds8", "ds9", "ds10",
]


# ------------------------------------------------------------------
# Training pipeline
# ------------------------------------------------------------------
def model_training_pipeline(
    training_data_path: str,
    testing_data_path: str,
    results_path: str,
    best_model_path: str,
    version: str,
    target_col: str,
) -> None:
    """
    Train models across multiple datasets, track experiments in MLflow,
    persist experiment results, and save the best-performing model.
    """
    LOGGER.info("Starting training pipeline | version=%s", version)
    mlflow.set_experiment(EXPERIMENT_NAME)

    all_results: Dict[str, Dict] = {}
    best_model: Optional[Dict] = None
    best_roc_auc: float = float("-inf")

    for dataset_name in DATASETS:
        LOGGER.info("Training on dataset: %s", dataset_name)

        with mlflow.start_run(run_name=f"{version}_{dataset_name}") as run:
            run_id = run.info.run_id

            # ------------------------------------------------------
            # Params
            # ------------------------------------------------------
            mlflow.log_params(
                {
                    "dataset": dataset_name,
                    "version": version,
                    "target_col": target_col,
                }
            )

            # ------------------------------------------------------
            # Load data
            # ------------------------------------------------------
            X_train, y_train = ingest_data(
                f"{training_data_path}{dataset_name}.csv",
                index_col="row_id",
                target_col=target_col,
            )
            X_test, y_test = ingest_data(
                f"{testing_data_path}{dataset_name}.csv",
                index_col="row_id",
                target_col=target_col,
            )

            _validate_data(X_train, X_test, dataset_name)

            # ------------------------------------------------------
            # Train & evaluate
            # ------------------------------------------------------
            experiment_result = experiments.experiment_results(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                version=version,
            )

            experiment_result.update(
                {
                    "dataset_name": dataset_name,
                    "mlflow_run_id": run_id,
                }
            )

            all_results[dataset_name] = experiment_result

            # ------------------------------------------------------
            # Metrics
            # ------------------------------------------------------
            _log_numeric_metrics(experiment_result)

            # ------------------------------------------------------
            # Model
            # ------------------------------------------------------
            model = experiment_result.get("best_estimator")
            if model is not None:
                mlflow.sklearn.log_model(
                    model,
                    name="model",
                )

            # ------------------------------------------------------
            # Best model tracking
            # ------------------------------------------------------
            roc_auc = experiment_result.get("test_roc_auc")
            if roc_auc is None:
                LOGGER.warning(
                    "Missing test_roc_auc | dataset=%s",
                    dataset_name,
                )
                continue

            if roc_auc > best_roc_auc:
                best_roc_auc = roc_auc
                best_model = experiment_result
                mlflow.set_tag("best_model", "true")

                LOGGER.info(
                    "New best model | dataset=%s | roc_auc=%.5f",
                    dataset_name,
                    roc_auc,
                )

    _persist_results(
        all_results,
        best_model,
        results_path,
        best_model_path,
        version,
    )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _validate_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    dataset_name: str,
) -> None:
    nan_rows = X_test[X_test.isna().any(axis=1)]
    if not nan_rows.empty:
        LOGGER.warning(
            "NaNs detected | dataset=%s | rows=%s",
            dataset_name,
            nan_rows.index.tolist(),
        )


def _log_numeric_metrics(results: Dict) -> None:
    for key, value in results.items():
        if isinstance(value, (int, float)):
            mlflow.log_metric(key, value)


def _persist_results(
    all_results: Dict[str, Dict],
    best_model: Optional[Dict],
    results_path: str,
    best_model_path: str,
    version: str,
) -> None:
    # --------------------------------------------------------------
    # Save experiment results
    # --------------------------------------------------------------
    results_file = (
        f"{results_path}{version}/model_experiment_results.pkl"
    )
    save_pickle(all_results, results_file)
    LOGGER.info("Saved experiment results | path=%s", results_file)

    # --------------------------------------------------------------
    # Save best model
    # --------------------------------------------------------------
    if best_model is None:
        LOGGER.error("No valid best model found")
        return

    model_file = (
        f"{best_model_path}{version}/"
        f"best_model_{best_model['dataset_name']}.pkl"
    )
    save_pickle(best_model, model_file)

    LOGGER.info(
        "Saved best model | dataset=%s | path=%s",
        best_model["dataset_name"],
        model_file,
    )

# ------------------------------------------------------------------
# Inference pipeline
# ------------------------------------------------------------------
def model_inference_pipeline(
    inference_data_path: str,
    results_path: str,
    version: str,
    selected_ds: str,
    threshold: float,
) -> None:
    """
    Run inference on new data using a previously trained model
    and export prediction probabilities.
    """
    LOGGER.info(
        "Starting inference | version=%s | dataset=%s",
        version,
        selected_ds,
    )

    # --------------------------------------------------------------
    # MLflow (same experiment as training)
    # --------------------------------------------------------------
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(
        run_name=f"inference_{version}_{selected_ds}"
    ):

        mlflow.log_params(
            {
                "version": version,
                "dataset": selected_ds,
                "threshold": threshold,
                "pipeline_stage": "inference",
            }
        )

        # ----------------------------------------------------------
        # Load data
        # ----------------------------------------------------------
        X_infer, _ = ingest_data(
            f"{inference_data_path}{selected_ds}.csv",
            index_col="row_id",
        )

        mlflow.log_metric("num_inference_rows", X_infer.shape[0])

        # ----------------------------------------------------------
        # Load model (unchanged)
        # ----------------------------------------------------------
        model_path = (
            f"{results_path}{version}/best_model_{selected_ds}.pkl"
        )
        model_dict = load_pickle(model_path)

        model = model_dict.get("best_estimator")
        if model is None:
            raise KeyError(
                "'best_estimator' not found in saved model dictionary"
            )

        LOGGER.debug("Loaded model from %s", model_path)

        # ----------------------------------------------------------
        # Prediction
        # ----------------------------------------------------------
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_infer)[:, 1]
        else:
            LOGGER.warning(
                "Model does not support predict_proba. Using predict()."
            )
            y_pred_proba = model.predict(X_infer)

        y_pred_bool = y_pred_proba >= threshold

        mlflow.log_metric(
            "positive_prediction_rate",
            float(y_pred_bool.mean()),
        )

        # ----------------------------------------------------------
        # Export results
        # ----------------------------------------------------------
        df_results = pd.DataFrame(
            {
                "prediction_proba": y_pred_proba,
                "prediction": y_pred_bool,
            },
            index=X_infer.index,
        )

        export_path = (
            f"{results_path}{version}/final_inferences.csv"
        )
        export_data(df_results, export_path)

        mlflow.log_artifact(export_path)

        LOGGER.info(
            "Inference completed successfully | output=%s",
            export_path,
        )
