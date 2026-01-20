import numpy as np

def apply_concept_drift(df, feature, target, flip_prob=0.2, random_state=None):
    """
    Changes relationship between feature and target.
    """
    rng = np.random.default_rng(random_state)
    df_drifted = df.copy()

    mask = df_drifted[feature] == df_drifted[feature].mode()[0]
    flip_mask = rng.random(mask.sum()) < flip_prob

    idx_to_flip = df_drifted.loc[mask].index[flip_mask]
    df_drifted.loc[idx_to_flip, target] = 1 - df_drifted.loc[idx_to_flip, target]

    return df_drifted

def apply_stronger_concept_drift(df, feature, target, flip_prob=0.5, random_state=None):
    """
    Introduces stronger concept drift by flipping target values for a higher proportion of rows.
    """
    rng = np.random.default_rng(random_state)
    df_drifted = df.copy()

    # Pick the most common feature value (mode)
    mode_val = df_drifted[feature].mode()[0]
    mask = df_drifted[feature] == mode_val

    # Flip a larger fraction of target values
    flip_mask = rng.random(mask.sum()) < flip_prob
    idx_to_flip = df_drifted.loc[mask].index[flip_mask]
    df_drifted.loc[idx_to_flip, target] = 1 - df_drifted.loc[idx_to_flip, target]

    return df_drifted