from src.drift.utils_drift import compute_distribution, kl_divergence, chi_squared_test
from src.drift.utils_drift import compute_distribution, kl_divergence

def prior_drift(df_old, df_new, columns=None, bins=10, threshold=0.1):
    """Detect drift in marginal distributions."""
    if columns is None:
        columns = df_old.columns
    drift_scores = {}
    for col in columns:
        p_old, _ = compute_distribution(df_old, col, bins=bins)
        p_new, _ = compute_distribution(df_new, col, bins=bins)
        score = kl_divergence(p_old, p_new)
        drift_scores[col] = {'kl_divergence': score, 'drift': score > threshold}
    return drift_scores

def prior_drift_categorical(df_old, df_new, columns=None, threshold=0.05):
    """
    Detect drift in categorical features using Chi-squared test.
    Returns features where p-value < threshold (significant drift).
    """
    if columns is None:
        columns = df_old.columns
    drift_scores = {}
    
    for col in columns:
        counts_old = df_old[col].value_counts()
        counts_new = df_new[col].value_counts()
        
        # Align indices
        all_categories = counts_old.index.union(counts_new.index)
        counts_old = counts_old.reindex(all_categories, fill_value=0)
        counts_new = counts_new.reindex(all_categories, fill_value=0)
        
        chi2, p = chi_squared_test(counts_new.values, counts_old.values)
        drift_scores[col] = {'chi2': chi2, 'p_value': p, 'drift': p < threshold}
    
    return drift_scores