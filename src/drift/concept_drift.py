from src.utils.storage import ingest_data
import pandas as pd
import numpy as np

# --------------------------
# Step 0: Generate drift dictionaries
# --------------------------
def generate_drift_dicts(df, target_column='target', feature_columns=None):
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
# Step 1: Simulate concept drift
# --------------------------
import pandas as pd
import numpy as np
from src.utils.storage import ingest_data
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
def simulate_drift_auto(df_path, target_column='target', feature_columns=None, random_state=42):
    # Load data
    X, y = ingest_data(df_path, index_col='row_id', target_col=target_column)
    
    # Combine X and y to single DataFrame
    df = pd.concat([X, y], axis=1)
    
    np.random.seed(random_state)
    
    if feature_columns is None:
        feature_columns = [c for c in df.columns if c != target_column]
    
    # Split dataset
    n = len(df)
    split_idx = n // 2
    df_pre = df.iloc[:split_idx].copy()
    df_post = df.iloc[split_idx:].copy()
    
    # Generate drift dicts
    feature_drift_dict, target_drift_dict = generate_drift_dicts(df_pre, target_column, feature_columns)
    
    # Apply drift to second half
    for col in feature_columns:
        categories = list(feature_drift_dict[col].keys())
        probs = list(feature_drift_dict[col].values())
        df_post[col] = np.random.choice(categories, size=len(df_post), p=probs)
        df_post[target_column] = df_post[col].apply(lambda x: int(np.random.rand() < target_drift_dict[col][x]))
    
    df_drifted = pd.concat([df_pre, df_post]).reset_index(drop=True)
    
    
    return df, df_drifted, feature_drift_dict, target_drift_dict
