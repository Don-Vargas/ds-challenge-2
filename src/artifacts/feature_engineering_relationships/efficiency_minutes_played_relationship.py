import numpy as np

# -------------------------------------------------
# 1. Efficiency Per Minute
# -------------------------------------------------

def add_efficiency_per_minute(df):
    """
    Add efficiency per minute feature.

    Calculates efficiency per minute played and handles division-by-zero
    and missing values by replacing infinities and NaNs with 0.
    """
    df = df.copy()
    df['eff_per_min'] = df['efficiency'] / df['minutes_played']
    df['eff_per_min'] = df['eff_per_min'].replace([np.inf, -np.inf], 0)
    df['eff_per_min'] = df['eff_per_min'].fillna(0)
    return df


# -------------------------------------------------
# 2. Total contribution
# -------------------------------------------------

def add_eff_times_minutes(df):
    """
    Add total contribution feature.

    Computes total contribution as efficiency multiplied by minutes played.
    """
    df = df.copy()
    df['eff_times_minutes'] = df['efficiency'] * df['minutes_played']
    return df

# -------------------------------------------------
# 3. High-intensity player flag 
# -------------------------------------------------
def add_high_eff_min(df):
    """
    Add high-intensity player flag.

    Flags players with high efficiency per minute and significant playing time.
    A player is flagged if:
        - efficiency per minute > 0.8
        - minutes played > 30
    """
    df = df.copy()
    df['high_eff_min'] = np.where(
        (df['eff_per_min'] > 0.8) & (df['minutes_played'] > 30), 1, 0
    )
    return df

# -------------------------------------------------
# Efficiency-Minutes_played Feature Engineering Pipeline
# -------------------------------------------------
def engineering_pipeline(df):
    """
    Applies all feature engineering steps to the DataFrame.
    """

    df = add_efficiency_per_minute(df)
    df = add_eff_times_minutes(df)
    df = add_high_eff_min(df)
    return df
