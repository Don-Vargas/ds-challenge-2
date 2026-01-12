import numpy as np

# -------------------------------------------------
# 1. Points Per Minute
# -------------------------------------------------
def add_points_per_min(df):
    """
    Add points per minute feature.

    Calculates scoring rate while handling division-by-zero
    and missing values by replacing infinities and NaNs with 0.
    """
    df = df.copy()
    df['points_per_min'] = df['points'] / df['minutes_played']
    df['points_per_min'] = df['points_per_min'].replace([np.inf, -np.inf], 0)
    df['points_per_min'] = df['points_per_min'].fillna(0)
    return df


# -------------------------------------------------
# 2. Scoring Volume
# -------------------------------------------------
def add_scoring_volume(df):
    """
    Add scoring volume feature.

    Computes total scoring volume as points multiplied by minutes played.
    """
    df = df.copy()
    df['scoring_volume'] = df['points'] * df['minutes_played']
    return df


# -------------------------------------------------
# 3. High Usage Scorer Flag
# -------------------------------------------------
def add_high_usage_scorer(df):
    """
    Add high-usage scorer flag.

    Flags players with high scoring output and heavy playing time.
    A player is flagged if:
        - points >= 20
        - minutes played >= 30
    """
    df = df.copy()
    df['high_usage_scorer'] = (
        (df['points'] >= 20) & (df['minutes_played'] >= 30)
        ).astype(int)
    return df

# -------------------------------------------------
# Points–Minutes Feature Engineering Pipeline
# -------------------------------------------------
def engineering_pipeline(df):
    """
    Applies all points–minutes feature engineering steps to the DataFrame.
    """
    df = add_points_per_min(df)
    df = add_scoring_volume(df)
    df = add_high_usage_scorer(df)

    return df
