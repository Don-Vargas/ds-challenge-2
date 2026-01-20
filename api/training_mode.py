from typing import List, Tuple

import src.preprocessing.pre_processing as pre_processing
import src.modeling.modeling as modeling

from config.staging import (
    MODEL_PARAMETER_RESULTS,
    TRAINING_DATA,
    TESTING_DATA,
)
from config.research import (
    RAW_DATA,
    TRAIN_DATA,
    TEST_DATA,
    SPLIT_SIZE,
    EDA_DATASET_PATH,
    MODELING_RESULTS,
)
from src.research import data_split


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------
def starter(run_split: bool = False, test_size: float = SPLIT_SIZE) -> None:
    if not run_split:
        return

    data_split.split_and_save_datasets(
        RAW_DATA,
        TRAIN_DATA,
        TEST_DATA,
        test_size=test_size,
        random_state=42,
    )


def start_training(
    version: str = "v1",
    target_col: str = "target",
    test_size: float = SPLIT_SIZE,
):
    starter(run_split=True, test_size=test_size)

    datasets = [
        ("train", TRAIN_DATA),
        ("test", TEST_DATA),
    ]

    for role, data_path in datasets:
        pre_processing.preprocessing_pipeline(
            data_path,
            EDA_DATASET_PATH,
            version,
            target_col=target_col,
            role=role,
        )

    modeling.model_training_pipeline(
        TRAINING_DATA,
        TESTING_DATA,
        MODELING_RESULTS,
        MODEL_PARAMETER_RESULTS,
        version,
        target_col=target_col,
    )

    return {
        "mode": "training",
        "version": version,
        "test_size": test_size,
        "status": "completed",
    }
