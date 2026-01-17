import logging
from typing import List, Tuple

import src.preprocessing.pre_processing as pre_processing
import src.modeling.modeling as modeling

from config.staging import (
    MODEL_PARAMETER_RESULTS,
    TRAINING_DATA,
    TESTING_DATA,
    MODEL_DATA_SET
)
from config.research import (
    RAW_DATA,
    TRAIN_DATA,
    TEST_DATA,
    SPLIT_SIZE,
    INFERENCE_DATA,
    EDA_REPORT_PATH,
    EDA_FIGURES_PATH,
    EDA_DATASET_PATH,
    MODELING_RESULTS
)
from src.research import data_split
from src.utils.storage import path_validate


# ------------------------------------------------------------------
# Logging configuration (entry-point verbosity)
# ------------------------------------------------------------------
LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | %(levelname)-8s | "
        "%(name)s | %(message)s"
    ),
)


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------
def starter(run_split: bool = False) -> None:
    """
    Optionally split raw data into train/test datasets.
    """
    if not run_split:
        LOGGER.info("Data split step skipped.")
        return

    LOGGER.info("Starting dataset split.")
    data_split.split_and_save_datasets(
        RAW_DATA,
        TRAIN_DATA,
        TEST_DATA,
        test_size=SPLIT_SIZE,
        random_state=42,
    )
    LOGGER.info("Dataset split completed successfully.")


# ------------------------------------------------------------------
# Main execution
# ------------------------------------------------------------------
if __name__ == "__main__":
    LOGGER.info("=" * 80)
    LOGGER.info("PIPELINE STARTED")
    LOGGER.info("=" * 80)

    inference_mode: bool = True # False to train
    version: str = "v1"
    target_col: str = "target"

    LOGGER.info("Run configuration:")
    LOGGER.info("  inference_mode : %s", inference_mode)
    LOGGER.info("  version        : %s", version)
    LOGGER.info("  target column  : %s", target_col)

    # --------------------------------------------------------------
    # Validate required output paths
    # --------------------------------------------------------------
    LOGGER.info("Validating required output paths.")
    routes: List[str] = [
        EDA_REPORT_PATH,
        EDA_FIGURES_PATH,
        EDA_DATASET_PATH,
    ]

    for route in routes:
        path_validate(route)
        LOGGER.debug("Validated path: %s", route)

    LOGGER.info("All required paths validated.")

    # --------------------------------------------------------------
    # Dataset configuration
    # --------------------------------------------------------------
    datasets: List[Tuple[str, str]] = [
        ("train", TRAIN_DATA),
        ("test", TEST_DATA),
    ]

    training_data_path: str = TRAINING_DATA
    testing_data_path: str = TESTING_DATA
    best_model_path: str = MODEL_PARAMETER_RESULTS
    threshold = 0.5

    # --------------------------------------------------------------
    # Inference mode
    # --------------------------------------------------------------
    if inference_mode:
        LOGGER.info("-" * 80)
        LOGGER.info("INFERENCE MODE ENABLED")
        LOGGER.info("-" * 80)

        role: str = "inference"
        results_path: str = MODEL_PARAMETER_RESULTS
        selected_ds: str = "ds4"

        LOGGER.info(
            "Running inference | dataset=%s | version=%s",
            selected_ds,
            version,
        )

        pre_processing.preprocessing_inference_pipeline(
            INFERENCE_DATA,
            results_path,
            version,
            selected_ds,
        )

        modeling.model_inference_pipeline(
            MODEL_DATA_SET,
            results_path,
            version,
            selected_ds,
            threshold,
        )

        LOGGER.info("Inference pipeline completed successfully.")

 # --------------------------------------------------------------
    # Training mode
    # --------------------------------------------------------------
    else:
        LOGGER.info("-" * 80)
        LOGGER.info("TRAINING MODE ENABLED")
        LOGGER.info("-" * 80)

        starter(run_split=True) # true to split

        LOGGER.info("Starting preprocessing pipelines.")
        for role, data_path in datasets:
            LOGGER.info(
                "Preprocessing | role=%s | path=%s",
                role,
                data_path,
            )

            pre_processing.preprocessing_pipeline(
                data_path,
                EDA_DATASET_PATH,
                version,
                target_col=target_col,
                role=role,
            )

        LOGGER.info("Preprocessing completed for all datasets.")

        LOGGER.info("Starting model training pipeline.")
        modeling.model_training_pipeline(
            training_data_path,
            testing_data_path,
            MODELING_RESULTS,
            best_model_path,
            version,
            target_col=target_col,
        )

        LOGGER.info("Model training pipeline completed.")

    LOGGER.info("=" * 80)
    LOGGER.info("PIPELINE FINISHED SUCCESSFULLY")
    LOGGER.info("=" * 80)
