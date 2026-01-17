import time
from sklearn.ensemble import RandomForestClassifier
import logging

logger = logging.getLogger(__name__)

def train_random_forest(X, y,max_d=3, n_estimators=500, random_state=42):
    logger.info(
        "Training RandomForest (%d estimators) on data with shape X=%s, y=%s",
        n_estimators, X.shape, y.shape
    )

    start = time.perf_counter()

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_d,
        random_state=random_state
    )
    rf.fit(X, y)

    elapsed = time.perf_counter() - start
    logger.info("RandomForest training completed in %.2f seconds", elapsed)

    return rf
