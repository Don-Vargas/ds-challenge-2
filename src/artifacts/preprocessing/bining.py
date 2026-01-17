import pandas as pd
import numpy as np
from collections import Counter
import pandas as pd
import numpy as np
import json

def standard_binning(column_df, n_bins=5, labels=None):
    """
    Standard equal-width binning with persistence of bin edges.
    
    Returns:
        binned_series: pd.Series with bin labels
        bin_config: dict with bin edges and labels
    """
    # Compute bin edges
    bin_edges = np.linspace(column_df.min(), column_df.max(), n_bins + 1)
    
    # Bin the data
    binned_series = pd.cut(column_df, bins=bin_edges, labels=labels, include_lowest=True)
    
    # Prepare configuration for persistence
    bin_config = {
        "bin_edges": bin_edges.tolist(),
        "labels": labels,
        "binning_type": "standard"
    }
    
    return binned_series, bin_config


def quantile_binning(column_df, n_bins=5, labels=None):
    """
    Quantile-based binning with persistence of computed quantile edges.
    
    Returns:
        binned_series: pd.Series with bin labels
        bin_config: dict with quantile bin edges and labels
    """
    # Compute bins and return edges
    binned_series, bin_edges = pd.qcut(column_df, q=n_bins, labels=labels, retbins=True, duplicates='drop')
    
    # Prepare configuration for persistence
    bin_config = {
        "bin_edges": bin_edges.tolist(),
        "labels": labels,
        "binning_type": "quantile"
    }
    
    return binned_series, bin_config


def apply_back_binning(column_df, bin_config):
    """
    Apply previously saved bin configuration to new data.
    
    Args:
        column_df: pd.Series of new data
        bin_config: dict from a previous binning step
        
    Returns:
        pd.Series with binned data
    """
    bin_edges = bin_config["bin_edges"]
    labels = bin_config.get("labels", None)
    # Use pd.cut with stored edges
    return pd.cut(column_df, bins=bin_edges, labels=labels, include_lowest=True), None

