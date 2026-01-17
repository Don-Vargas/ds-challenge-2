import time
import pandas as pd
from sklearn.inspection import permutation_importance

import logging

logger = logging.getLogger(__name__)

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
