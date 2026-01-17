from sklearn.decomposition import PCA
import pandas as pd

def pca_by_variance(X, variance_explain=0.80):
    """
    Reduce dimensionality while keeping `variance_explain` explained variance.
    """
    pca = PCA(n_components=variance_explain)
    X_reduced = pca.fit_transform(X)

    # Create DataFrame with original index
    component_names = [f"PC{i+1}" for i in range(pca.n_components_)]
    X_reduced_df = pd.DataFrame(
        X_reduced,
        index=X.index,
        columns=component_names
    )

    return {
        'X_reduced': X_reduced_df,
        'pca_config': pca
    }


def pca_transform(X, pca_config):
    """
    Transform new data using a fitted PCA and return a DataFrame
    similar to pca_by_variance.
    """
    pca = pca_config["pca"]
    X_reduced = pca.transform(X)

    # Create DataFrame with original index and component names
    component_names = [f"PC{i+1}" for i in range(pca.n_components_)]
    X_reduced_df = pd.DataFrame(
        X_reduced,
        index=X.index,
        columns=component_names
    )

    return {
        'X_reduced': X_reduced_df,
        'pca_config': None  # No need to return the config again
    }
