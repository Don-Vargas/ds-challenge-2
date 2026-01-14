import pandas as pd

def frequency_encoding(column_df):
    """
    Performs frequency encoding for a given column in a DataFrame.
    Returns only the new encoded column, not the original one.
    """
    freq = column_df.value_counts() / len(column_df)
    encoded = column_df.map(freq)
    encoding_config = {
        "mapping": freq.to_dict(),
        "encoding_type": "frequency"
    }
    
    return encoded, encoding_config

def apply_back_frequency_encoding(column_df, encoding_config):
    """
    Applies a precomputed frequency encoding to a column using the provided encoding_config.
    Unseen values will be encoded as 0.
    """
    mapping = encoding_config["mapping"]
    
    encoded = column_df.map(mapping).fillna(0)
    
    return encoded, None

def one_hot_encoding(column_df):
    """
    One-hot encoding for all columns in a DataFrame.
    Returns a DataFrame with one-hot encoded columns for all columns in the input DataFrame,
    excluding the original columns.
    """
    encoded_df = pd.DataFrame()
    for col in column_df.columns:
        encoded_col = pd.get_dummies(column_df[col], prefix=col, drop_first=True)
        encoded_df = pd.concat([encoded_df, encoded_col], axis=1)
    
    return encoded_df
