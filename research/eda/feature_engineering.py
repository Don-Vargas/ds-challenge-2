import pandas as pd
from src.artifacts import encoders
from src.artifacts import scalers

def engineering(df_path):
    df = pd.read_csv(df_path, index_col='row_id')

    # Frequency encoding
    freq_cols = ['position', 'team', 'opponent']
    for col in freq_cols:
        df[f'{col}_freq'] = encoders.frequency_encoding(df[col])

    # One-hot encoding
    onehot_cols = ['game_location', 'steals', 'blocks', 'turnovers']
    for col in onehot_cols:
        df[f'{col}_onehot'] = encoders.one_hot_encoding(df[col])

    # Standardization
    standard_cols = ['minutes_played', 'fg_pct', 'three_pct', 'ft_pct']
    for col in standard_cols:
        df[f'{col}_standard'] = scalers.standardization(df[col])

    # MinMaxation
    standard_cols = ['minutes_played', 'fg_pct', 'three_pct', 'ft_pct']
    for col in standard_cols:
        df[f'{col}_standard'] = scalers.standardization(df[col])















    df = pd.concat([df, game_location_encoded], axis=1)
    return df
