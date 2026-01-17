import logging
from typing import Dict, Any

import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier


# ---------------------------------------------------
# Logging configuration
# ---------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# ---------------------------------------------------
# Model configuration (ROC-AUC only)
# ---------------------------------------------------
MODEL_TRAINING_CONFIGS: Dict[str, Dict[str, Any]] = {
    "decision_tree": {
        "model": DecisionTreeClassifier(),
        "params": {
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2],
        },
    },
    "random_forest": {
        "model": RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
        ),
        "params": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2],
        },
    },
    "gradient_boosting": {
        "model": GradientBoostingClassifier(random_state=42),
        "params": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [3, 5],
        },
    },
}


def experiment_results(
    X_train,
    y_train,
    X_test,
    y_test,
    version: str,
) -> Dict[str, Any]:
    """Train models using ROC-AUC optimization and compare results."""
    logger.info("Starting experiment version: %s", version)

    # ---------------------------------------------------
    # CV setup
    # ---------------------------------------------------
    cv = StratifiedKFold(
        n_splits=10,
        shuffle=True,
        random_state=42,
    )

    roc_results: Dict[str, Dict[str, Any]] = {}
    best_overall: Dict[str, Any] = {
        "model_name": None,
        "estimator": None,
        "test_roc_auc": float("-inf"),
    }

    # ---------------------------------------------------
    # Training + ROC-AUC Optimization
    # ---------------------------------------------------
    for model_name, config in MODEL_TRAINING_CONFIGS.items():
        logger.info("Optimizing ROC-AUC for model: %s", model_name)

        pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", config["model"]),
            ]
        )

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=config["params"],
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            verbose=0,
        )

        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        y_test_proba = best_model.predict_proba(X_test)[:, 1]
        test_roc_auc = roc_auc_score(y_test, y_test_proba)

        roc_results[model_name] = {
            "cv_best_roc_auc": grid_search.best_score_,
            "test_roc_auc": test_roc_auc,
            "best_params": grid_search.best_params_,
        }

        logger.info(
            "Model: %s | CV ROC-AUC: %.4f | Test ROC-AUC: %.4f",
            model_name,
            grid_search.best_score_,
            test_roc_auc,
        )

        if test_roc_auc > best_overall["test_roc_auc"]:
            best_overall.update(
                {
                    "model_name": model_name,
                    "estimator": best_model,
                    "test_roc_auc": test_roc_auc,
                    "cv_best_roc_auc": grid_search.best_score_,
                    "best_params": grid_search.best_params_,
                }
            )

    # ---------------------------------------------------
    # ROC-AUC comparison summary
    # ---------------------------------------------------
    roc_auc_df = (
        pd.DataFrame.from_dict(roc_results, orient="index")
        .sort_values("test_roc_auc", ascending=False)
    )

    logger.info("ROC-AUC model comparison:\n%s", roc_auc_df)

    return {
        "version": version,
        "best_model_name": best_overall["model_name"],
        "best_estimator": best_overall["estimator"],
        "best_params": best_overall["best_params"],
        "cv_best_roc_auc": best_overall["cv_best_roc_auc"],
        "test_roc_auc": best_overall["test_roc_auc"],
        "all_model_results": roc_results,
    }
