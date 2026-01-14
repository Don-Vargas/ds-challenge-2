import pandas as pd
import time
import logging

from src.artifacts.feature_importance.importance_ranking import get_tree_importance, get_permutation_importance
from src.artifacts.feature_importance.importance_models import train_random_forest

logger = logging.getLogger(__name__)

# --------------------------
# 
# --------------------------

def rank_all_features(datasets, y):
    all_rankings = {} 

    for ds, X in datasets.items():
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

        all_rankings[ds] = {
            "tree_importance": tree_imp,
            "permutation_importance": perm_imp,
            "aggregated_ranking": df
        }

    return all_rankings

