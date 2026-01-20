import numpy as np
import pandas as pd


def apply_covariate_drift(df, feature, drift_strength=0.3, random_state=None):
    """
    Shifts the distribution of a binned feature.
    drift_strength ∈ [0,1]
    """
    rng = np.random.default_rng(random_state)

    values = df[feature].values
    unique_bins = df[feature].unique()

    n_drift = int(len(df) * drift_strength)
    drift_idx = rng.choice(len(df), n_drift, replace=False)

    df_drifted = df.copy()
    df_drifted.loc[drift_idx, feature] = rng.choice(unique_bins, size=n_drift)

    return df_drifted


