import logging

from src.evidently.evidently_binary import (
    run_evidently_binary_classification,
)

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# ------------------------------------------------------------------
# Driver
# ------------------------------------------------------------------
if __name__ == "__main__":
    run_evidently_binary_classification(
        version="v2",
        dataset_name="ds4",
        index_col="row_id",
        target_col="target",
        training_data_path="src/research/dataset/train/ds4.csv",
        testing_data_path="src/research/dataset/test/ds4.csv",
        results_path="inference_results/",
        artifacts_dir="evidently/",
    )
