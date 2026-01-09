import pandas as pd

def frequency_encoding(column_df):
    """
    Performs frequency encoding for a given column in a DataFrame.
    """
    freq = column_df.value_counts() / len(column_df)
    encoded = column_df.map(freq)
    return encoded

def one_hot_encoding(column_df, prefix=None):
    """
    One-hot encoding for a single column.
    Returns a DataFrame with one-hot columns.
    """
    if prefix is None:
        prefix = column_df.name  # Use column name as prefix
    encoded_df = pd.get_dummies(column_df, prefix=prefix)
    return encoded_df
