import pandas as pd
import time
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
from src.artifacts.feature_importance.importance_models import train_random_forest
from src.utils.storage import path_validate

import logging

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid")  # nicer plots


# --------------------------
# Functions for feature importance
# --------------------------
def get_tree_importance(model, X):
    logger.info("Computing tree-based feature importance")
    start = time.perf_counter()

    importance = pd.Series(model.feature_importances_, index=X.columns)

    elapsed = time.perf_counter() - start
    logger.info("Tree importance computed in %.2f seconds", elapsed)

    return importance.sort_values(ascending=False)


def get_permutation_importance(model, X, y, n_repeats=10, random_state=42):
    logger.info(
        "Computing permutation importance (n_repeats=%d)", n_repeats
    )

    start = time.perf_counter()

    result = permutation_importance(
        model, X, y,
        n_repeats=n_repeats,
        random_state=random_state
    )

    elapsed = time.perf_counter() - start
    logger.info("Permutation importance computed in %.2f seconds", elapsed)

    importance = pd.Series(result.importances_mean, index=X.columns)
    return importance.sort_values(ascending=False)


def rank_features(X, y):
    logger.info("Ranking features (%d total features)", X.shape[1])
    start_total = time.perf_counter()

    model = train_random_forest(X, y)

    tree_imp = get_tree_importance(model, X)
    perm_imp = get_permutation_importance(model, X, y)

    df = pd.DataFrame({
        'tree': tree_imp,
        'permutation': perm_imp
    })

    df['avg_rank'] = df.rank(ascending=False).mean(axis=1)
    df = df.sort_values('avg_rank')

    elapsed_total = time.perf_counter() - start_total
    logger.info(
        "Feature ranking completed in %.2f seconds. Top feature: %s",
        elapsed_total, df.index[0]
    )

    return {
        "tree_importance": tree_imp,
        "permutation_importance": perm_imp,
        "aggregated_ranking": df
    }

# --------------------------
# Plotting function
# --------------------------

def plot_feature_importance(feature_rankings, 
                            top_n=10, dataset_name="Dataset", 
                            output_path='output_path', 
                            dataset='dataset'):
    """
    Plots the top_n features for Tree, Permutation, and Aggregated importance.
    """
    logger.info(
        "Generating feature importance plots for %s (top_n=%d)",
        dataset_name, top_n
    )

    importance_types = ['tree_importance', 'permutation_importance']

    for imp_type in importance_types:
        logger.info("Plotting %s for dataset %s", imp_type, dataset_name)

        imp = feature_rankings[imp_type].head(top_n)
        plt.figure(figsize=(16, 5))
        sns.barplot(
            x=imp.values,
            y=imp.index,
            hue=imp.index,
            palette="viridis",
            legend=False
        )
        plt.tight_layout()

        file_path = f"{output_path}feature_importance_importance/{dataset}/{imp_type}.png"
        path_validate(file_path)

        plt.savefig(file_path, dpi=300)
        plt.close()

        logger.info("Saved plot to %s", file_path)

    # Aggregated ranking
    agg = feature_rankings['aggregated_ranking'].head(top_n)
    plt.figure(figsize=(16, 5))
    sns.barplot(
        x=agg['avg_rank'],
        y=agg.index,
        hue=agg.index,
        palette="magma",
        legend=False
    )
    plt.tight_layout()

    file_path = f"{output_path}feature_importance_importance/{dataset}/aggregated_ranking.png"
    path_validate(file_path)

    plt.savefig(file_path, dpi=300)
    plt.close()

    logger.info("Saved aggregated ranking plot to %s", file_path)
