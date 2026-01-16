import pre_processing
import modeling

from config.staging import (MODEL_PARAMETER_RESULTS, TRAINING_DATA, TESTING_DATA, MODEL_DATA_SET)
from config.research import (
    RAW_DATA, TRAIN_DATA, TEST_DATA, SPLIT_SIZE, INFERENCE_DATA,
    EDA_REPORT_PATH, EDA_FIGURES_PATH, EDA_DATASET_PATH, MODELING_RESULTS
)
from src.utils.storage import path_validate
from src.research.eda import data_split

def starter(run_split: bool = False):
    if run_split:
        data_split.split_and_save_datasets(
            RAW_DATA,
            TRAIN_DATA, TEST_DATA,
            test_size=SPLIT_SIZE, random_state=42
        )

if __name__ == "__main__":
    inference_mode = 1
    version = 'v1'
    target_col = 'target'

    routes = [EDA_REPORT_PATH, EDA_FIGURES_PATH, EDA_DATASET_PATH]
    for route in routes:
        path_validate(route)

    # Define datasets and their roles
    datasets = [('train', TRAIN_DATA), ('test', TEST_DATA)]
    training_data_path = TRAINING_DATA
    testing_data_path  = TESTING_DATA
    best_model_path = MODEL_PARAMETER_RESULTS

    if inference_mode == 0:
        starter()

        for role, data_path in datasets:
            pre_processing.preprocessing_pipeline(
                data_path, 
                EDA_DATASET_PATH, 
                version, 
                target_col=target_col, 
                role=role
            )
        modeling.model_training_pipeline(training_data_path, testing_data_path, MODELING_RESULTS, best_model_path, version, target_col=target_col)

    elif inference_mode == 1:
        role = 'inference'
        results_path = MODEL_PARAMETER_RESULTS
        selected_ds = 'ds4'
        pre_processing.preprocessing_inference_pipeline(INFERENCE_DATA, results_path, version, selected_ds)
        modeling.model_inference_pipeline(MODEL_DATA_SET, results_path, version, selected_ds)
