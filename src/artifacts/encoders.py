import pandas as pd

def frequency_encoding_train(column_df):
    """
    Performs frequency encoding for a training column.
    Returns the encoded column and the mapping (frequency dictionary).
    """
    freq = column_df.value_counts() / len(column_df)
    encoded = column_df.map(freq)
    return encoded, freq.to_dict()

def frequency_encoding_test(column_df, freq_mapping):
    """
    Applies frequency encoding to a test column using a mapping from training.
    Unseen categories are encoded as 0 (or you can choose another value).
    """
    encoded = column_df.map(freq_mapping).fillna(0)
    return encoded

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
