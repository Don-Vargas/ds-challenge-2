from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def pca_by_variance(X, variance_explain = 0.80):
    """
    Reduce dimensionality while keeping 80% explained variance.
    """
    # Standardize (important for PCA)
    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=variance_explain)
    X_reduced = pca.fit_transform(X_scaled)

    return X_reduced, pca.n_components_, pca.explained_variance_ratio_
