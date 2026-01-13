import pandas as pd
import time
import logging
from src.artifacts.feature_importance.importance_ranking import rank_features, plot_feature_importance

logger = logging.getLogger(__name__)

# --------------------------
# Main function
# --------------------------

def ranking_kings(datasets_path, output_path):
    datasets = ['_ds1', '_ds3', '_ds5']

    logger.info(
        "Starting feature importance ranking for %d datasets",
        len(datasets)
    )

    pipeline_start = time.perf_counter()

    for ds in datasets:
        dataset_start = time.perf_counter()
        logger.info("Processing dataset: %s", ds)

        df = pd.read_csv(
            f'{datasets_path}{ds}.csv',
            index_col='row_id'
        )

        logger.info("Loaded dataset %s with shape %s", ds, df.shape)

        X = df.drop(columns=['target', 'player_id'])
        X.columns = X.columns.astype(str)
        y = df['target']

        feature_rankings = rank_features(X, y)

        plot_feature_importance(
            feature_rankings,
            top_n=10,
            dataset_name=ds,
            output_path=output_path,
            dataset=ds
        )

        elapsed_ds = time.perf_counter() - dataset_start
        logger.info(
            "Dataset %s processed in %.2f seconds",
            ds, elapsed_ds
        )

    total_elapsed = time.perf_counter() - pipeline_start
    logger.info(
        "Feature importance ranking pipeline completed in %.2f seconds",
        total_elapsed
    )
