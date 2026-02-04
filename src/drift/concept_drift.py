from src.utils.storage import ingest_data
from config.research import CURRENT_DATA
import src.preprocessing.pre_processing as pre_processing
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import mlflow
from typing import List, Optional, Tuple, Dict

# --------------------------
# Generate drift dictionaries
# --------------------------
def generate_drift_dicts(
    df: pd.DataFrame,
    target_column: str = 'target',
    feature_columns: Optional[List[str]] = None
) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    """
    Generate feature_drift and target_drift dictionaries for a dataframe.
    """
    if feature_columns is None:
        feature_columns = [c for c in df.columns if c != target_column]

    feature_drift = {}
    target_drift = {}

    for col in feature_columns:
        # Feature distribution (normalized counts)
        feature_drift[col] = df[col].value_counts(normalize=True).to_dict()
        # Target probability per category
        target_drift[col] = df.groupby(col)[target_column].mean().to_dict()
    
    return feature_drift, target_drift


# --------------------------
# Simulate concept drift
# --------------------------
def simulate_drift_auto(
        df_path, version='v1', 
        selected_ds='ds4', 
        feature_columns=None, 
        target_column='target', 
        random_state=42,
        ):

    pre_processing.preprocessing_inference_pipeline(
        CURRENT_DATA,
        df_path,
        version,
        selected_ds,
    )
    # Load data
    _, target = ingest_data(CURRENT_DATA, index_col='row_id', target_col=target_column)
    df_path = f"{df_path}inference/{selected_ds}.csv"
    X, _ = ingest_data(df_path, index_col='row_id')
    
    # Combine X and y
    df = pd.concat([X, target], axis=1)
    
    np.random.seed(random_state)
    
    if feature_columns is None:
        feature_columns = [c for c in df.columns if c != target_column]
    
    # Split dataset
    n = len(df)
    split_idx = n // 2
    df_pre = df.iloc[:split_idx].copy()
    df_post = df.iloc[split_idx:].copy()
    
    # Generate reference (pre-drift) dicts
    feature_drift_pre, target_drift_pre = generate_drift_dicts(df_pre, target_column, feature_columns)
    
    # Apply drift to second half
    for col in feature_columns:
        categories = list(feature_drift_pre[col].keys())
        probs = list(feature_drift_pre[col].values())
        df_post[col] = np.random.choice(categories, size=len(df_post), p=probs)
        df_post[target_column] = df_post[col].apply(lambda x: int(np.random.rand() < target_drift_pre[col][x]))
    
    # Generate current (post-drift) dicts
    feature_drift_post, target_drift_post = generate_drift_dicts(df_post, target_column, feature_columns)
    
    df_drifted = pd.concat([df_pre, df_post]).reset_index(drop=True)
    
    return df, df_drifted, feature_drift_pre, feature_drift_post, target_drift_pre, target_drift_post
