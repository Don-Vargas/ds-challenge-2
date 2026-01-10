import pandas as pd
from src.artifacts import encoders, scalers, bining

def engineering(df_path, df_version):
    df = pd.read_csv(df_path, index_col='row_id')
    df_new = pd.DataFrame(index=df.index)
    df_new = pd.concat([df_new, df.target, df.player_id], axis=1)
    ds1 = ds2 = ds3 = ds4 = ds5 = ds6 = df_new.copy()

    # Frequency encoding
    cols = ['position', 'team', 'opponent',
                 'game_location', 'rest_days']
    for col in cols:
        new_col = f'{col}_freq'
        df_new[new_col] = encoders.frequency_encoding(df[col])
        ds1[new_col] = ds2[new_col] = df_new[new_col]

    ds3[cols] = ds4[cols] = df[cols]
    print(ds3.columns)

    cols = ['steals', 'blocks', 'turnovers']
    ds3[cols] = ds4[cols] = df[cols]

    standard_norm_cols = ['minutes_played', 'fg_pct', 'three_pct',  
                          'ft_pct', 'age', 'plus_minus', 
                          'efficiency', 'points', 'rebounds', 
                          'assists', 'steals', 'blocks', 
                          'turnovers']
    for col in standard_norm_cols:
        # Standardization
        new_col = f'{col}_standard'
        df_new[new_col] = scalers.standardization(df[col])
        ds1[new_col] = ds2[new_col] = df_new[new_col]
        # MinMaxation
        new_col = f'{col}_minmax'
        df_new[f'{col}_minmax'] = scalers.minimaxation(df[col])
        ds1[new_col] = ds2[new_col] = df_new[new_col]

    bining_cols = [('minutes_played', 5), ('fg_pct', 8), ('three_pct', 5), 
                   ('ft_pct', 6), ('age', 9), ('plus_minus', 7), 
                   ('efficiency', 20), ('points', 6), ('rebounds', 7), 
                   ('assists', 4)]
    for col, n_bins in bining_cols:
        # Binning standard
        new_col = f'{col}_binning_standard'
        df_new[new_col] = bining.standard_binning(df[col], n_bins=n_bins, labels=list(range(1, n_bins + 1)))
        ds3[new_col] = ds4[new_col] = df_new[new_col]
        # Binning quantile binning
        new_col = f'{col}_binning_quantile'
        df_new[new_col] = bining.quantile_binning(df[col], n_bins=n_bins, labels=list(range(1, n_bins + 1)))
        ds3[new_col] = ds4[new_col] = df_new[new_col]

    # One-hot encoding
    ds3_ohe_cols = list(set(ds3.columns) - set(['target', 'player_id']))
    ds4_ohe_cols = list(set(ds4.columns) - set(['target', 'player_id']))
    ds5 = pd.concat([ds5, encoders.one_hot_encoding(ds3[ds3_ohe_cols])], axis=1)
    ds6 = pd.concat([ds6, encoders.one_hot_encoding(ds4[ds4_ohe_cols])], axis=1)

    # Save reports
    datasets = [ds1, ds2, ds3, ds4, ds5, ds6]
    dataset_names = ['ds1', 'ds2', 'ds3', 'ds4', 'ds5', 'ds6']

    # Ordenamos los datasets
    [ds.sort_values(by=['player_id'], inplace=True) for ds in datasets]

    # Guardamos los archivos CSV con los nombres correspondientes
    for ds, name in zip(datasets, dataset_names):
        ds.to_csv(f'{df_version}_{name}.csv', index=True)
