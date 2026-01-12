import numpy as np

# -------------------------------------------------
# 1. Efficiency Per Point
# -------------------------------------------------
def add_eff_per_point(df):
    """
    Add efficiency per point feature.

    Calculates efficiency per point scored and handles division-by-zero
    and missing values by replacing infinities and NaNs with 0.
    """
    df = df.copy()
    df['eff_per_point'] = df['efficiency'] / df['points']
    df['eff_per_point'] = df['eff_per_point'].replace([np.inf, -np.inf], 0)
    df['eff_per_point'] = df['eff_per_point'].fillna(0)
    return df


# -------------------------------------------------
# 2. Scoring Impact
# -------------------------------------------------
def add_scoring_impact(df):
    """
    Add scoring impact feature.

    Computes scoring impact as efficiency multiplied by points scored.
    """
    df = df.copy()
    df['scoring_impact'] = df['efficiency'] * df['points']
    return df


# -------------------------------------------------
# 3. High-Efficiency Scorer Flag
# -------------------------------------------------
def add_high_eff_scorer_flag(df):
    """
    Add high-efficiency scorer flag.

    Flags players who are both efficient and high scorers.
    A player is flagged if:
        - efficiency >= 20
        - points >= 15
    """
    df = df.copy()
    df['high_eff_scorer'] = (
        (df['efficiency'] >= 20) & (df['points'] >= 15)
        ).astype(int)
    return df


# -------------------------------------------------
# Efficiency-Points Feature Engineering Pipeline
# -------------------------------------------------
def engineering_pipeline(df):
    """
    Applies all efficiency–points feature engineering steps to the DataFrame.
    """
    df = add_eff_per_point(df)
    df = add_scoring_impact(df)
    df = add_high_eff_scorer_flag(df)

    return df
