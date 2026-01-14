from collections import defaultdict
import logging
from config.research import (
    RAW_DATA, TRAIN_DATA, TEST_DATA, SPLIT_SIZE, INFERENCE_DATA,
    EDA_REPORT_PATH, EDA_FIGURES_PATH, EDA_DATASET_PATH
)
from src.utils.storage import (
    path_validate,
    ingest_data,
    export_data,
    save_pickle,
    load_pickle
)
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

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("preprocessing_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def starter(run_split: bool = False):
    if run_split:
        logger.info("Splitting raw dataset into train and test sets.")
        data_split.split_and_save_datasets(
            RAW_DATA,
            TRAIN_DATA, TEST_DATA,
            test_size=SPLIT_SIZE, random_state=42
        )
        logger.info(f"Datasets saved: TRAIN_DATA={TRAIN_DATA}, TEST_DATA={TEST_DATA}")

# --- Column Definitions ---
FREQ_ENCODING_COLS = [
    'position', 'team', 'opponent', 'game_location', 'rest_days',
    'high_usage_scorer', 'high_eff_min', 'high_eff_scorer'
]

RAW_NUMERIC_COLS = ['steals', 'blocks', 'turnovers']

SCALING_COLS = [
    'minutes_played', 'fg_pct', 'three_pct', 'ft_pct', 'age',
    'plus_minus', 'efficiency', 'points', 'rebounds',
    'assists', 'steals', 'blocks', 'turnovers',
    'eff_per_point', 'eff_per_min', 'points_per_min',
    'scoring_impact', 'eff_times_minutes', 'scoring_volume'
]

BINNING_COLS = [
    ('minutes_played', 5), ('fg_pct', 8), ('three_pct', 5),
    ('ft_pct', 6), ('age', 9), ('plus_minus', 7),
    ('efficiency', 20), ('points', 6), ('rebounds', 7),
    ('assists', 4), ('eff_per_point', 6), ('eff_per_min', 6),
    ('points_per_min', 6), ('scoring_impact', 6),
    ('eff_times_minutes', 6), ('scoring_volume', 6)
]

DS_KEYS = {
    "ds1": {"frequency_encoding": FREQ_ENCODING_COLS, "scaling": {"standard": SCALING_COLS}},
    "ds2": {"frequency_encoding": FREQ_ENCODING_COLS, "scaling": {"minmax": SCALING_COLS}},
    "ds3": {"frequency_encoding": [c for c in FREQ_ENCODING_COLS if c != 'rest_days'],
            "raw_numeric": RAW_NUMERIC_COLS,
            "binning": {"standard": BINNING_COLS}},
    "ds4": {"frequency_encoding": [c for c in FREQ_ENCODING_COLS if c != 'rest_days'],
            "raw_numeric": RAW_NUMERIC_COLS,
            "binning": {"quantile": BINNING_COLS}},
    "ds5": {"one_hot_from": "ds3"},
    "ds6": {"one_hot_from": "ds4"},
    "ds7": {"pca_from": "ds1"},
    "ds8": {"pca_from": "ds2"},
    "ds9": {"pca_from": "ds3"},
    "ds10": {"pca_from": "ds4"}
}

def init_datasets(df, ds_keys):
    logger.info(f"Initializing {len(ds_keys)} datasets copies.")
    return {ds: df.copy(deep=True) for ds in ds_keys}

def preprocessing_pipeline(data_path, results_path, version='last_version', target_col=None, role='train'):
    logger.info(f"Starting preprocessing pipeline. Role: {role}, Version: {version}")
    processing_configs_file = f'training_parameter_results/{version}/processing_configs.pkl'
    all_rankings_file = f'training_parameter_results/{version}/all_rankings.pkl'

    if role != 'train':
        logger.info("Loading processing configurations and rankings for inference.")
        processing_configs = load_pickle(processing_configs_file)
        all_rankings = load_pickle(all_rankings_file)
    else:
        processing_configs = defaultdict(dict)
    
    logger.info(f"Ingesting data from {data_path}")
    X, y = ingest_data(data_path, index_col='row_id', target_col=target_col)
    player_id = X['player_id']
    X = X.drop(columns=['player_id'])
    logger.info(f"Data ingested. Shape: {X.shape}")

    logger.info("Starting feature engineering pipeline.")
    X = feature_engineering.feature_creation_pipeline(X)

    ds = init_datasets(X, DS_KEYS)
    ds, processing_configs = dataset_engineering.feature_engineering_pipeline(ds, DS_KEYS, processing_configs=processing_configs, role=role)

    if role == 'train':
        logger.info("Ranking all features for training datasets.")
        all_rankings = feature_importance.rank_all_features(ds, y, DS_KEYS)
        save_pickle(processing_configs, processing_configs_file)
        save_pickle(all_rankings, all_rankings_file)
        logger.info(f"Processing configurations and rankings saved in 'training_parameter_results/{version}'")
        ds = training_dataset_building.dataset_building(ds, all_rankings)
        export_results = f"{results_path}train/{name}.csv"
    else:
        export_results = f"{results_path}test/{name}.csv"

    logger.info(f"Exporting processed datasets to {results_path}")
    for name in ds:
        export_data(ds[name], export_results)
        logger.info(f"Dataset {name} exported successfully.")

if __name__ == "__main__":
    role = 'train'
    #role = 'test'
    version = 'v1'
    target_col = 'target'
    results_path = EDA_DATASET_PATH

    routes = [EDA_REPORT_PATH, EDA_FIGURES_PATH, EDA_DATASET_PATH]
    for route in routes:
        path_validate(route)
        logger.info(f"Validated path: {route}")
    if role == 'train':
        data_path = TRAIN_DATA
    elif role == 'test':
        data_path = TEST_DATA
    else:
        data_path = INFERENCE_DATA

    logger.info("Starting pipeline execution.")
    starter()
    preprocessing_pipeline(data_path, results_path, version, target_col=target_col, role=role)
    logger.info("Pipeline execution completed.")
