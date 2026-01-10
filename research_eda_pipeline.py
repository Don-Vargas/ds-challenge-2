from research.eda import (data_split, 
                          eda, 
                          plot_distributions, 
                          feature_engineering, 
                          correlation_collinearity)
from utils.storage import path_validate
from config.research import (
    RAW_DATA,TRAIN_DATA,TEST_DATA,SPLIT_SIZE,
    EDA_REPORT_PATH, EDA_FIGURES_PATH, EDA_DATASET_PATH
)

if __name__ == "__main__":
    ds_version = 'v1'
    routes = [EDA_REPORT_PATH, EDA_FIGURES_PATH, EDA_DATASET_PATH]
    [path_validate(route) for route in routes]
    data_split.split_and_save_datasets(RAW_DATA,TRAIN_DATA,TEST_DATA, test_size=SPLIT_SIZE, random_state=42)
    eda.explore_dataset(TRAIN_DATA, EDA_REPORT_PATH + 'eda_simple.csv', EDA_REPORT_PATH + 'eda.csv', include_categorical = True)
    plot_distributions.generate_plots(TRAIN_DATA, EDA_FIGURES_PATH)
    feature_engineering.engineering(TRAIN_DATA, f'{EDA_DATASET_PATH}_{ds_version}')
    #correlation_collinearity.test_features(f'{EDA_DATASET_PATH}_{ds_version}')
