from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


# --------------------------------------------------
# Utility: evaluation (binary classification)
# --------------------------------------------------
def evaluate_model(model, X, y, threshold=0.5):
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)

    return {
        "roc_auc": roc_auc_score(y, proba),
    }


# --------------------------------------------------
# GridSearch runner (generic)
# --------------------------------------------------
def run_grid_search(
    model,
    param_grid,
    X_train,
    y_train,
    cv=5,
    n_jobs=-1,
):
    """
    Runs GridSearchCV using ROC-AUC for model selection.
    """

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring={
            "roc_auc": "roc_auc",
        },
        refit="roc_auc",
        cv=cv,
        n_jobs=n_jobs,
        verbose=1,
    )

    grid.fit(X_train, y_train)

    return grid


# --------------------------------------------------
# Decision Tree GridSearch
# --------------------------------------------------
def gridsearch_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier(
        class_weight="balanced",
        random_state=42,
    )

    param_grid = {
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 10, 50],
        "min_samples_leaf": [1, 5, 20],
        "criterion": ["gini", "entropy"],
    }

    return run_grid_search(model, param_grid, X_train, y_train)


# --------------------------------------------------
# Random Forest GridSearch
# --------------------------------------------------
def gridsearch_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    param_grid = {
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 10],
        "min_samples_leaf": [1, 5],
        "max_features": ["sqrt", "log2"],
    }

    return run_grid_search(model, param_grid, X_train, y_train)


# --------------------------------------------------
# Gradient Boosting GridSearch
# --------------------------------------------------
def gridsearch_gradient_boosting(X_train, y_train):
    model = GradientBoostingClassifier(random_state=42)

    param_grid = {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5],
        "subsample": [0.8, 1.0],
    }

    return run_grid_search(model, param_grid, X_train, y_train)


# --------------------------------------------------
# Full experiment runner
# --------------------------------------------------
def run_all_tree_experiments(
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        threshold=0.5,
    ):
    results = {}

    searches = {
        "decision_tree": gridsearch_decision_tree(X_train, y_train),
        "random_forest": gridsearch_random_forest(X_train, y_train),
        "gradient_boosting": gridsearch_gradient_boosting(X_train, y_train),
    }

    for name, search in searches.items():
        best_model = search.best_estimator_

        metrics = {
            "cv_best_roc_auc": search.best_score_,
            "best_params": search.best_params_,
            "train": evaluate_model(best_model, X_train, y_train, threshold),
        }

        if X_val is not None and y_val is not None:
            metrics["val"] = evaluate_model(best_model, X_val, y_val, threshold)

        results[name] = {
            "model": best_model,
            "metrics": metrics,
        }

    return results



def plot_model_roc_auc(results, dataset="Validation"):
    """
    Plots model names vs ROC-AUC scores.

    Parameters
    ----------
    results : dict
        Dictionary returned by `run_all_tree_experiments`.
        Expected format:
        {
            "decision_tree": {"metrics": {"train": {...}, "val": {...}, ...}, "model": ...},
            ...
        }
    dataset : str
        Which dataset to plot: "train" or "val" (case insensitive)
    """
    dataset = dataset.lower()
    if dataset not in ["train", "val"]:
        raise ValueError("dataset must be 'train' or 'val'")

    model_names = []
    roc_aucs = []

    for model_name, data in results.items():
        model_names.append(model_name)
        metrics = data["metrics"]
        if dataset not in metrics:
            raise ValueError(f"No '{dataset}' metrics found for model '{model_name}'")
        roc_aucs.append(metrics[dataset]["roc_auc"])

    # Plot
    plt.figure(figsize=(8, 5))
    bars = plt.bar(model_names, roc_aucs, color="skyblue")
    plt.ylim(0, 1)
    plt.ylabel("ROC-AUC")
    plt.title(f"Model ROC-AUC ({dataset.capitalize()} set)")

    # Add value labels on top of bars
    for bar, value in zip(bars, roc_aucs):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.show()
