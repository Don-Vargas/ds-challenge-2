from collections import defaultdict
import logging

from src.utils.storage import (
    ingest_data,
    export_data,
    save_pickle,
    load_pickle
)
from src.research import (
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
    logger.info(f"Initializing {len(ds_keys)} dataset copies.")
    return {ds: df.copy(deep=True) for ds in ds_keys}


def preprocessing_pipeline(data_path, results_path, version='last_version',
                           target_col=None, role='train'):
    logger.info("=" * 80)
    logger.info(
        f"Starting preprocessing pipeline | Role: {role} | "
        f"Version: {version} | Target column: {target_col}"
    )

    processing_configs_file = f'training_parameter_results/{version}/processing_configs.pkl'
    all_rankings_file = f'training_parameter_results/{version}/all_rankings.pkl'

    # ------------------------------------------------------------------
    # Load or initialize processing configurations
    # ------------------------------------------------------------------
    if role != 'train':
        logger.info("Inference mode detected.")
        logger.info(f"Loading processing configs from: {processing_configs_file}")
        logger.info(f"Loading feature rankings from: {all_rankings_file}")

        processing_configs = load_pickle(processing_configs_file)
        all_rankings = load_pickle(all_rankings_file)

        logger.info(
            f"Loaded processing configs with {len(processing_configs)} feature groups "
            f"and rankings for {len(all_rankings)} datasets."
        )
    else:
        logger.info("Training mode detected. Initializing empty processing configurations.")
        processing_configs = defaultdict(dict)

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------
    logger.info(f"Ingesting raw data from: {data_path}")
    X, y = ingest_data(data_path, index_col='row_id', target_col=target_col)

    logger.info(
        f"Raw data ingested successfully | "
        f"Features shape: {X.shape} | "
        f"Target present: {y is not None}"
    )

    player_id = X['player_id']
    X = X.drop(columns=['player_id'])
    logger.info("Dropped identifier column: 'player_id'")

    # ------------------------------------------------------------------
    # Feature creation
    # ------------------------------------------------------------------
    logger.info("Starting feature creation pipeline.")
    X = feature_engineering.feature_creation_pipeline(X)
    logger.info(f"Feature creation completed. New feature matrix shape: {X.shape}")

    # ------------------------------------------------------------------
    # Dataset initialization
    # ------------------------------------------------------------------
    logger.info(f"Initializing datasets with keys: {DS_KEYS}")
    ds = init_datasets(X, DS_KEYS)

    logger.info(
        "Datasets initialized: "
        + ", ".join(f"{k}({v.shape})" for k, v in ds.items())
    )

    # ------------------------------------------------------------------
    # Dataset-level feature engineering
    # ------------------------------------------------------------------
    logger.info("Starting dataset-level feature engineering.")
    ds, processing_configs = dataset_engineering.feature_engineering_pipeline(
        ds,
        DS_KEYS,
        processing_configs=processing_configs,
        role=role
    )

    logger.info(
        "Dataset-level feature engineering completed. "
        "Updated dataset shapes: "
        + ", ".join(f"{k}({v.shape})" for k, v in ds.items())
    )

    # ------------------------------------------------------------------
    # Feature ranking (training only)
    # ------------------------------------------------------------------
    if role == 'train':
        logger.info("Ranking all features for training datasets.")
        all_rankings = feature_importance.rank_all_features(ds, y, DS_KEYS)

        logger.info("Feature ranking completed. Saving artifacts.")
        save_pickle(processing_configs, processing_configs_file)
        save_pickle(all_rankings, all_rankings_file)

        logger.info(
            f"Processing configurations and rankings saved to "
            f"'training_parameter_results/{version}'"
        )

    # ------------------------------------------------------------------
    # Dataset building
    # ------------------------------------------------------------------
    logger.info("Building final training/inference datasets.")
    ds = training_dataset_building.dataset_building(ds, all_rankings, y, role=role)

    logger.info(
        "Final dataset building completed. "
        + ", ".join(f"{k}({v.shape})" for k, v in ds.items())
    )

    # ------------------------------------------------------------------
    # Export results
    # ------------------------------------------------------------------
    logger.info(f"Exporting processed datasets to: {results_path}{role}/")
    for name in ds:
        export_path = f"{results_path}{role}/{name}.csv"
        export_data(ds[name], export_path)
        logger.info(f"Dataset '{name}' exported successfully to {export_path}")

    logger.info("Preprocessing pipeline completed successfully.")
    logger.info("=" * 80)


def preprocessing_inference_pipeline(data_path, results_path, version='last_version',
                                     selected_ds='ds1'):
    """
    Preprocess new/blind data for inference using a selected dataset configuration.

    Args:
        data_path (str): Path to raw data CSV.
        results_path (str): Directory to save processed datasets.
        version (str): Version folder where processing configs are stored.
        selected_ds (str): Dataset key from DS_KEYS to process.
    """
    logger.info("=" * 80)
    logger.info(
        f"Starting inference preprocessing | Version: {version} | "
        f"Selected dataset: {selected_ds}"
    )

    processing_configs_file = f'training_parameter_results/{version}/processing_configs.pkl'
    all_rankings_file = f'training_parameter_results/{version}/all_rankings.pkl'

    logger.info("Loading processing configurations and feature rankings from training.")
    processing_configs = load_pickle(processing_configs_file)
    all_rankings = load_pickle(all_rankings_file)
    all_rankings = {selected_ds: all_rankings[selected_ds]}

    logger.info(f"Ingesting blind data from: {data_path}")
    X, y = ingest_data(data_path, index_col='row_id')

    X = X.drop(columns=['player_id'])
    logger.info("Dropped identifier column: 'player_id'")

    logger.info("Starting feature creation pipeline for inference data.")
    X = feature_engineering.feature_creation_pipeline(X)
    logger.info(f"Feature creation completed. Feature matrix shape: {X.shape}")

    ds = {selected_ds: X.copy(deep=True)}
    logger.info(f"Initialized inference dataset: {selected_ds}({ds[selected_ds].shape})")

    logger.info("Starting dataset-level feature engineering for inference.")
    ds, _ = dataset_engineering.feature_engineering_pipeline(
        ds,
        {selected_ds: DS_KEYS[selected_ds]},
        processing_configs=processing_configs,
        role='inference'
    )
    logger.info(f"Dataset-level feature engineering completed for {selected_ds}.")

    logger.info("Building final inference dataset using stored feature rankings.")
    ds = training_dataset_building.dataset_building(ds, all_rankings, y, role='inference')
    logger.info(f"Final dataset built for {selected_ds}. Shape: {ds[selected_ds].shape}")

    export_path = f"{results_path}inference/{selected_ds}.csv"
    export_data(ds[selected_ds], export_path)
    logger.info(f"Inference dataset '{selected_ds}' exported successfully to {export_path}")
    logger.info("Inference preprocessing pipeline completed successfully.")
    logger.info("=" * 80)

    return ds[selected_ds]
