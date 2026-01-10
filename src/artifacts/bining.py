import pandas as pd
import numpy as np
from collections import Counter

def standard_binning(column_df, n_bins=5, labels=None):
    """
    Standard equal-width binning (default).
    """
    return pd.cut(column_df, bins=n_bins, labels=labels)

def quantile_binning(column_df, n_bins=5, labels=None):
    """
    Quantile-based binning (distribution-aware).
    - Creates bins so that each bin has roughly the same number of samples.
    - Dense regions get smaller bins automatically.
    """
    return pd.qcut(column_df, q=n_bins, labels=labels, duplicates='drop')
