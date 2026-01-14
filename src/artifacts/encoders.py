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

def one_hot_encoding(df, drop_first=True):
    """
    One-hot encode all columns in a DataFrame and return:
      1. The encoded DataFrame
      2. Metadata needed to decode back later (categories, dropped category, prefix)
    
    Parameters:
    - df: pd.DataFrame, input categorical columns
    - drop_first: bool, whether to drop the first category to avoid multicollinearity
    
    Returns:
    - encoded_df: pd.DataFrame with one-hot columns
    - one_hot_config: dict with metadata for each column
    """
    one_hot_config = {}
    
    encoded_df = pd.DataFrame(index=df.index)
    
    for col in df.columns:
        # Get categories
        categories = df[col].astype(str).unique().tolist()
        dropped = categories[0] if drop_first else None
        
        # Save metadata
        one_hot_config[col] = {
            "categories": categories,
            "dropped": dropped,
            "prefix": col
        }
        
        # Perform one-hot encoding for this column
        col_encoded = pd.get_dummies(df[col].astype(str), prefix=col, drop_first=drop_first)
        encoded_df = pd.concat([encoded_df, col_encoded], axis=1)
    
    return encoded_df, one_hot_config


def apply_back_one_hot_encoding(df, one_hot_config):
    encoded_df = pd.DataFrame(index=df.index)
    
    for col, info in one_hot_config.items():
        categories = info["categories"]
        dropped = info["dropped"]
        prefix = info["prefix"]
        
        # create all possible one-hot columns
        col_encoded = pd.get_dummies(df[col].astype(str), prefix=prefix)
        for cat in categories:
            if cat == dropped:
                continue
            col_name = f"{prefix}_{cat}"
            if col_name not in col_encoded:
                # create missing columns as 0
                col_encoded[col_name] = 0
                
        # ensure columns are in the same order as in training
        final_cols = [f"{prefix}_{c}" for c in categories if c != dropped]
        encoded_df = pd.concat([encoded_df, col_encoded[final_cols]], axis=1)
    
    return encoded_df
