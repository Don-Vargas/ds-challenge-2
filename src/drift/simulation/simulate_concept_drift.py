import pandas as pd
from src.utils.storage import ingest_data
from src.drift.concept_drift import apply_concept_drift, apply_stronger_concept_drift
from src.drift.utils_drift import detect_feature_drift


def concept_drift_simulation(
    file_path: str,
    features,
    target,
    alpha=0.05,
    return_data=False
):
    """
    Simulates concept drift and runs chi-square detection on features.
    """
    X, y = ingest_data(file_path, index_col='row_id', target_col=target)

    reference_df = pd.concat([X, y], axis=1)
    drifted_df = reference_df.copy()

    for feature in features:
        drifted_df = apply_stronger_concept_drift(
            drifted_df,
            feature,
            target
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

    if return_data:
        return reference_df, drifted_df, pd.DataFrame(drift_results)

    return drifted_df, pd.DataFrame(drift_results)
