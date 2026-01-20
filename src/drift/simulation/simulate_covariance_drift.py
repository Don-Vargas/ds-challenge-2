import pandas as pd
from covariance_drift import apply_covariate_drift
from utils_drift import detect_feature_drift


def simulate_covariate_drift(
    df,
    features,
    drift_strength=0.3,
    alpha=0.05
):
    """
    Simulates covariate drift and runs chi-square detection.
    """
    reference_df = df.copy()
    drifted_df = reference_df.copy()

    for feature in features:
        drifted_df = apply_covariate_drift(
            drifted_df,
            feature,
            drift_strength=drift_strength
        )

    drift_results = []
    for feature in features:
        result = detect_feature_drift(
            reference_df,
            drifted_df,
            feature,
            alpha
        )
        drift_results.append(result)

    return drifted_df, pd.DataFrame(drift_results)
