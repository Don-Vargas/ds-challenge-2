import logging
from config.research import (
    EDA_DATASET_PATH
)
from src.utils.storage import path_validate
from src.research.experiments import (
    load_data,
    tree_models
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    try:
        logging.info("Starting the data processing pipeline.")

        ds_version = 'v1'
        routes = [EDA_DATASET_PATH]

        for file in EDA_DATASET_PATH:

            X_train, y_train = load_data.data_loading(file)
            X_val, y_val = load_data.data_loading(file)

            # Run experiments (example)
            results = tree_models.run_all_tree_experiments(X_train, y_train, X_val, y_val)

            # Plot validation ROC-AUC
            tree_models.plot_model_roc_auc(results, dataset="val")

            # Plot training ROC-AUC
            tree_models.plot_model_roc_auc(results, dataset="train")


        logging.info("Data processing pipeline finished successfully.")

    except Exception as e:
        logging.exception(f"An error occurred: {e}")
