import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

def get_feature_distribution(df, feature, target=None):
    """
    Returns frequency distribution of a feature.
    If target is provided, returns conditional distribution P(feature | target).
    """
    if target is None:
        return df[feature].value_counts(normalize=True)
    else:
        return (
            df.groupby([target, feature])
              .size()
              .unstack(fill_value=0)
        )


def chi_square_drift_test(ref_counts, cur_counts):
    """
    Performs chi-square test between two categorical distributions.
    Returns statistic and p-value.
    """
    # Align bins
    ref_counts, cur_counts = ref_counts.align(cur_counts, fill_value=0)

    contingency_table = np.vstack([ref_counts.values, cur_counts.values])

    chi2, p_value, _, _ = chi2_contingency(contingency_table)
    return chi2, p_value


def detect_feature_drift(reference_df, current_df, feature, alpha=0.05):
    """
    Detects drift for a single feature using chi-square test.
    """
    ref_dist = get_feature_distribution(reference_df, feature)
    cur_dist = get_feature_distribution(current_df, feature)

    chi2, p_value = chi_square_drift_test(ref_dist, cur_dist)

    return {
        "feature": feature,
        "chi2": chi2,
        "p_value": p_value,
        "drift_detected": p_value < alpha
    }





'''

import numpy as np
from scipy.stats import chi2_contingency


def compute_mean_std(df, columns=None):
    """Compute mean and standard deviation for selected columns."""
    if columns is None:
        columns = df.columns
    stats = df[columns].agg(['mean', 'std']).T
    return stats

def compute_distribution(df, column, bins=10):
    """Compute histogram distribution for a column."""
    counts, edges = np.histogram(df[column], bins=bins, density=True)
    return counts, edges

def kl_divergence(p, q, eps=1e-10):
    """Compute KL divergence between two distributions."""
    p = np.asarray(p) + eps
    q = np.asarray(q) + eps
    return np.sum(p * np.log(p / q))

def chi_squared_test(obs_counts, exp_counts):
    """Perform chi-squared test for categorical distributions."""
    # Build a contingency table
    contingency_table = np.array([obs_counts, exp_counts])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    return chi2, p
'''