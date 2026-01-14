from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def pca_by_variance(X, variance_explain=0.80):
    """
    Reduce dimensionality while keeping `variance_explain` explained variance.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=variance_explain)
    X_reduced = pca.fit_transform(X_scaled)

    pca_config = {
        "scaler": scaler,
        "pca": pca
    }

    return (
        X_reduced,
        pca.n_components_,
        pca.explained_variance_ratio_,
        pca_config
    )

def pca_transform(X, pca_config):
    scaler = pca_config["scaler"]
    pca = pca_config["pca"]
    X_scaled = scaler.transform(X)
    return pca.transform(X_scaled)
