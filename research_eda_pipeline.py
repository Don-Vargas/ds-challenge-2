import logging
from config.research import (
    RAW_DATA, TRAIN_DATA, TEST_DATA, SPLIT_SIZE,
    EDA_REPORT_PATH, EDA_FIGURES_PATH, EDA_DATASET_PATH
)
from src.utils.storage import path_validate
from src.research.eda import (
    data_split,
    eda, 
    plot_distributions,
    correlation,
    feature_engineering,
    dataset_engineering,
    feature_importance,
    training_dataset_building
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
        routes = [EDA_REPORT_PATH, EDA_FIGURES_PATH, EDA_DATASET_PATH]

        logging.info("Validating paths...")
        for route in routes:
            path_validate(route)
            logging.info(f"Path validated: {route}")

        logging.info("Splitting dataset into train and test sets...")
        data_split.split_and_save_datasets(
            RAW_DATA, TRAIN_DATA, TEST_DATA, test_size=SPLIT_SIZE, random_state=42
        )
        logging.info(f"Datasets saved: {TRAIN_DATA}, {TEST_DATA}")

        logging.info("Performing initial EDA...")
        eda.explore_dataset(
            TRAIN_DATA,
            EDA_REPORT_PATH + 'eda_simple.csv',
            EDA_REPORT_PATH + 'eda.csv',
            include_categorical=True
        )
        logging.info("Initial EDA completed.")

        logging.info("Generating distribution plots...")
        plot_distributions.generate_plots(TRAIN_DATA, EDA_FIGURES_PATH)
        logging.info("Distribution plots generated.")

        logging.info("Running feature engineering pipeline...")
        feature_engineering_path = feature_engineering.feature_engineering_pipeline(
            TRAIN_DATA, f'{EDA_DATASET_PATH}_{ds_version}'
        )
        logging.info(f"Feature engineered dataset saved at: {feature_engineering_path}")

        logging.info("Performing EDA on engineered dataset...")
        eda.explore_dataset(
            feature_engineering_path,
            EDA_REPORT_PATH + 'eda_engineered_ds.csv',
            EDA_REPORT_PATH + 'eda.csv',
            include_categorical=True
        )
        logging.info("EDA on engineered dataset completed.")

        logging.info("Generating distribution plots for engineered dataset...")
        plot_distributions.generate_plots(feature_engineering_path, EDA_FIGURES_PATH)
        logging.info("Plots for engineered dataset generated.")

        logging.info("Running dataset engineering...")
        dataset_engineering.engineering(feature_engineering_path, f'{EDA_DATASET_PATH}_{ds_version}')
        logging.info("Dataset engineering completed.")

        logging.info("Testing feature correlations...")
        correlation.test_features(f'{EDA_DATASET_PATH}_{ds_version}', EDA_FIGURES_PATH)
        logging.info("Feature correlation testing completed.")

        logging.info("Calculating feature importance rankings...")
        feature_importance.ranking_kings(f'{EDA_DATASET_PATH}_{ds_version}', output_path=EDA_FIGURES_PATH)
        logging.info("Feature importance ranking completed.")

        training_dataset_building.dataset_building(f'{EDA_DATASET_PATH}_{ds_version}')

        logging.info("Data processing pipeline finished successfully.")

    except Exception as e:
        logging.exception(f"An error occurred: {e}")
