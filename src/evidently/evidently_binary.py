import logging
import pandas as pd
import mlflow

from src.utils.storage import ingest_data, load_pickle, path_validate
from src.modeling.modeling import EXPERIMENT_NAME

from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataQualityPreset, DataDriftPreset, TargetDriftPreset, ClassificationPreset


# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _validate_data(
    X_test: pd.DataFrame,
    dataset_name: str,
) -> None:
    """
    Log a warning if NaNs are detected in test data.
    """
    nan_rows = X_test[X_test.isna().any(axis=1)]
    if not nan_rows.empty:
        LOGGER.warning(
            "NaNs detected | dataset=%s | rows=%s",
            dataset_name,
            nan_rows.index.tolist(),
        )


def _build_binary_classification_frames(
    X: pd.DataFrame,
    y: pd.Series,
    model,
    target_col: str,
) -> pd.DataFrame:
    """
    Build Evidently-compatible dataframe for binary classification.
    Uses predicted probabilities for the positive class.
    """
    df = X.copy()
    df[target_col] = y

    if not hasattr(model, "predict_proba"):
        raise TypeError(
            "Binary classification requires predict_proba() for Evidently metrics"
        )

    df["prediction"] = model.predict_proba(X)[:, 1]
    return df


# ------------------------------------------------------------------
# Main Entry
# ------------------------------------------------------------------
def run_evidently_binary_classification(
    *,
    version: str,
    dataset_name: str,
    index_col: str,
    target_col: str,
    training_data_path: str,
    testing_data_path: str,
    results_path: str = "inference_results/",
    artifacts_dir: str = "evidently/",
) -> None:
    """
    Run an Evidently report focused on binary classification
    and log the HTML report as an MLflow artifact.
    """

    # --------------------------------------------------------------
    # Load model
    # --------------------------------------------------------------
    model_path = f"{results_path}{version}/best_model_{dataset_name}.pkl"
    model_dict = load_pickle(model_path)

    model = model_dict.get("best_estimator")
    if model is None:
        raise KeyError("'best_estimator' not found in saved model dictionary")

    LOGGER.info("Loaded model from %s", model_path)

    # --------------------------------------------------------------
    # MLflow
    # --------------------------------------------------------------
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"{version}_{dataset_name}") as run:
        run_id = run.info.run_id

        mlflow.log_params(
            {
                "dataset": dataset_name,
                "version": version,
                "target_col": target_col,
                "model_type": type(model).__name__,
                "task": "binary_classification",
            }
        )

        # ----------------------------------------------------------
        # Load data
        # ----------------------------------------------------------
        X_train, y_train = ingest_data(
            training_data_path,
            index_col,
            target_col,
        )

        X_test, y_test = ingest_data(
            testing_data_path,
            index_col,
            target_col,
        )

        _validate_data(X_test, dataset_name)

        # ----------------------------------------------------------
        # Build Evidently datasets
        # ----------------------------------------------------------
        train_df = _build_binary_classification_frames(
            X=X_train,
            y=y_train,
            model=model,
            target_col=target_col,
        )

        test_df = _build_binary_classification_frames(
            X=X_test,
            y=y_test,
            model=model,
            target_col=target_col,
        )

        # ----------------------------------------------------------
        # Evidently report (binary-only)
        # ----------------------------------------------------------
        LOGGER.info("Running Evidently binary classification report...")

        report = Report(
            metrics=[
                DataQualityPreset(),
                DataDriftPreset(),
                TargetDriftPreset(),
                ClassificationPreset(),
            ]
        )

        report.run(
            reference_data=train_df,
            current_data=test_df,
        )

        # ----------------------------------------------------------
        # Save + log report
        # ----------------------------------------------------------
        report_path = f"{artifacts_dir}evidently_binary_{version}_{dataset_name}.html"
        path_validate(report_path)

        report.save_html(report_path)
        mlflow.log_artifact(report_path, artifact_path="evidently")

        LOGGER.info(
            "Evidently binary classification report logged | run_id=%s",
            run_id,
        )
