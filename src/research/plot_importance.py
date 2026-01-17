import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os

sns.set_theme(style="whitegrid")  # nicer plots

# --------------------------
# Logging setup
# --------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --------------------------
# Helper function to validate path
# --------------------------
def path_validate(file_path):
    """Ensure the directory exists before saving the file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

# --------------------------
# Plotting function
# --------------------------
def plot_feature_importance(feature_rankings, 
                            top_n=10, 
                            dataset_name="Dataset", 
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
            dodge=False,
            legend=False
        )
        plt.tight_layout()

        file_path = f"{output_path}{dataset_name}/{imp_type}.png"
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
        dodge=False,
        legend=False
    )
    plt.tight_layout()

    file_path = f"{output_path}{dataset_name}/aggregated_ranking.png"
    path_validate(file_path)

    plt.savefig(file_path, dpi=300)
    plt.close()

    logger.info("Saved aggregated ranking plot to %s", file_path)
