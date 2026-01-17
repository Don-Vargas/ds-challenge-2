import logging
from typing import Dict, List, Optional

import pandas as pd

from src.model_experiments import experiments
from src.utils.storage import (
    ingest_data,
    export_data,
    save_pickle,
    load_pickle,
)


# ------------------------------------------------------------------
# Logging configuration
# ------------------------------------------------------------------
LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


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
    Train models across multiple datasets, store results,
    and persist the best-performing model.
    """
    LOGGER.info("Starting model training pipeline | version=%s", version)

    results: Dict[str, Dict] = {}
    best_model: Optional[Dict] = None
    best_roc_auc: float = float("-inf")

    for name in DATASETS:
        LOGGER.info("Processing dataset: %s", name)

        X_train, y_train = ingest_data(
            f"{training_data_path}{name}.csv",
            index_col="row_id",
            target_col=target_col,
        )
        X_test, y_test = ingest_data(
            f"{testing_data_path}{name}.csv",
            index_col="row_id",
            target_col=target_col,
        )

        rows_with_nans = X_test[X_test.isna().any(axis=1)]
        if not rows_with_nans.empty:
            LOGGER.warning(
                "NaNs detected in test set | dataset=%s | rows=%s",
                name,
                rows_with_nans.index.tolist(),
            )

        experiment_result = experiments.experiment_results(
            X_train,
            y_train,
            X_test,
            y_test,
            version,
        )
        experiment_result["dataset_name"] = name
        results[name] = experiment_result

        roc_auc = experiment_result.get("test_roc_auc")
        if roc_auc is None:
            LOGGER.warning(
                "Missing test_roc_auc in results | dataset=%s",
                name,
            )
            continue

        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            best_model = experiment_result
            LOGGER.info(
                "New best model found | dataset=%s | roc_auc=%.5f",
                name,
                roc_auc,
            )

    # --------------------------------------------------------------
    # Persist results
    # --------------------------------------------------------------
    results_export_path = (
        f"{results_path}{version}/model_experiment_results.pkl"
    )
    save_pickle(results, results_export_path)
    LOGGER.info("Saved experiment results to %s", results_export_path)

    if best_model is None:
        LOGGER.error("No valid model found. Best model was not saved.")
        return

    best_model_export_path = (
        f"{best_model_path}{version}/"
        f"best_model_{best_model['dataset_name']}.pkl"
    )
    save_pickle(best_model, best_model_export_path)
    LOGGER.info(
        "Saved best model | dataset=%s | path=%s",
        best_model["dataset_name"],
        best_model_export_path,
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

    X_infer, _ = ingest_data(
        f"{inference_data_path}{selected_ds}.csv",
        index_col="row_id",
    )

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

    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_infer)[:, 1]
    else:
        LOGGER.warning(
            "Model does not support predict_proba. "
            "Using predict() instead."
        )
        y_pred_proba = model.predict(X_infer)

    y_pred_bool = y_pred_proba >= threshold


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

    LOGGER.info(
        "Inference completed successfully | output=%s",
        export_path,
    )
