import pandas as pd
import numpy as np

# -------------------------------------------------
# 1. Efficiency per Point
# -------------------------------------------------
def add_eff_per_point(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes efficiency per point, handling division by zero.
    """
    df = df.copy()
    df['eff_per_point'] = df['efficiency'] / df['points']
    df['eff_per_point'] = df['eff_per_point'].replace([np.inf, -np.inf], 0)
    df['eff_per_point'] = df['eff_per_point'].fillna(0)
    return df

# -------------------------------------------------
# 2. Efficiency per Minute Played
# -------------------------------------------------
def add_eff_per_min(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes efficiency per minute played, handling division by zero.
    """
    df = df.copy()
    df['eff_per_min'] = df['efficiency'] / df['minutes_played']
    df['eff_per_min'] = df['eff_per_min'].replace([np.inf, -np.inf], 0)
    df['eff_per_min'] = df['eff_per_min'].fillna(0)
    return df

# -------------------------------------------------
# 3. Points per Minute Played
# -------------------------------------------------
def add_points_per_min(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes points per minute played, handling division by zero.
    """
    df = df.copy()
    df['points_per_min'] = df['points'] / df['minutes_played']
    df['points_per_min'] = df['points_per_min'].replace([np.inf, -np.inf], 0)
    df['points_per_min'] = df['points_per_min'].fillna(0)
    return df

# -------------------------------------------------
# 4. Combined Engineering Pipeline
# -------------------------------------------------
def engineering_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies all efficiency and points-related feature engineering steps.
    """
    df = add_eff_per_point(df)
    df = add_eff_per_min(df)
    df = add_points_per_min(df)
    return df
