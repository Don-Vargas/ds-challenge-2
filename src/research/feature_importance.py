import pandas as pd
import time
import logging
from src.artifacts.feature_importance.importance_ranking import get_tree_importance, get_permutation_importance
from src.artifacts.feature_importance.importance_models import train_random_forest

# --------------------------
# Configure logging
# --------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# --------------------------
# Feature ranking with top N selection and logging
# --------------------------
def rank_all_features(datasets, y, ds_keys, random_state=42, top_n=6):
    all_rankings = {} 
    pca_mapping = [ds for ds, cfg in ds_keys.items() if "pca_from" not in cfg]
    
    total_datasets = len(pca_mapping)
    logger.info(f"Starting feature ranking for {total_datasets} datasets.")

    for idx, dataset_name in enumerate(pca_mapping, start=1):
        start_time = time.time()
        X = datasets[dataset_name]

        logger.info(f"[{idx}/{total_datasets}] Processing dataset '{dataset_name}' with {X.shape[0]} samples and {X.shape[1]} features.")

        # Train model
        logger.info("  -> Training Random Forest model...")
        model = train_random_forest(X, y, random_state=random_state)
        logger.info("     Model trained.")

        # Get feature importance
        logger.info("  -> Calculating tree-based importance...")
        tree_importance = get_tree_importance(model, X)
        logger.info("     Tree importance computed.")

        logger.info("  -> Calculating permutation importance...")
        permutation_importance = get_permutation_importance(model, X, y, random_state=random_state)
        logger.info("     Permutation importance computed.")

        # Aggregate importances
        logger.info("  -> Aggregating importance rankings...")
        importance_features = {
            "tree": tree_importance,
            "permutation": permutation_importance,
        }
        importance_df = pd.DataFrame(importance_features, index=X.columns)
        importance_df["avg_rank"] = importance_df.rank(ascending=False).mean(axis=1)
        importance_df = importance_df.sort_values("avg_rank")

        # Select top N features
        top_features = importance_df.head(top_n).index.tolist()
        logger.info(f"  -> Top {top_n} features: {top_features}")

        # Store results
        all_rankings[dataset_name] = {
            "random_state": random_state,
            "features": list(X.columns),
            "tree_importance": tree_importance,
            "permutation_importance": permutation_importance,
            "aggregated_ranking": importance_df,
            "top_features": top_features, 
        }

        elapsed = time.time() - start_time
        logger.info(f"Completed dataset '{dataset_name}' in {elapsed:.2f} seconds.\n{'-'*40}")

    logger.info("Feature ranking completed for all datasets.")
    return all_rankings
