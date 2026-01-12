import pandas as pd
from src.artifacts import encoders, scalers, bining


def engineering(df_path, df_version):
    df = pd.read_csv(df_path, index_col='row_id')

    # Base dataframe
    base_df = pd.concat(
        [pd.DataFrame(index=df.index), df.target, df.player_id],
        axis=1
    )

    # Create independent datasets
    datasets = {f'ds{i}': base_df.copy(deep=True) for i in range(1, 7)}

    # ------------------------------------------------------------------
    # Frequency encoding
    # ------------------------------------------------------------------
    freq_cols = ['position', 'team', 'opponent', 'game_location', 'rest_days', 
                 'high_usage_scorer', 'high_eff_min', 'high_eff_scorer']

    for col in freq_cols:
        new_col = f'{col}_freq'
        freq_values = encoders.frequency_encoding(df[col])

        for k in ['ds1', 'ds2', 'ds3', 'ds4']:
            datasets[k][new_col] = freq_values

    # Raw categorical + discrete features for ds3 and ds4
    #for k in ['ds3', 'ds4']:
    #        datasets[k][freq_cols] = df[freq_cols]

    num_cols = ['steals', 'blocks', 'turnovers']
    for k in ['ds3', 'ds4']:
            datasets[k][num_cols] = df[num_cols]

    # ------------------------------------------------------------------
    # Standardization & MinMax scaling
    # ------------------------------------------------------------------
    standard_norm_cols = [
        'minutes_played', 'fg_pct', 'three_pct', 'ft_pct', 'age',
        'plus_minus', 'efficiency', 'points', 'rebounds',
        'assists', 'steals', 'blocks', 'turnovers',
        'eff_per_point', 'eff_per_min', 'points_per_min', 
        'scoring_impact', 'eff_times_minutes', 'scoring_volume'
    ]

    for col in standard_norm_cols:
        # Standard
        std_col = f'{col}_standard'
        std_values = scalers.standardization(df[col])
        datasets['ds1'][std_col] = std_values

        # MinMax
        mm_col = f'{col}_minmax'
        mm_values = scalers.minimaxation(df[col])
        datasets['ds2'][mm_col] = mm_values

    # ------------------------------------------------------------------
    # Binning
    # ------------------------------------------------------------------
    bining_cols = [
        ('minutes_played', 5), ('fg_pct', 8), ('three_pct', 5),
        ('ft_pct', 6), ('age', 9), ('plus_minus', 7),
        ('efficiency', 20), ('points', 6), ('rebounds', 7),
        ('assists', 4), ('eff_per_point', 6), ('eff_per_min', 6), 
        ('points_per_min', 6), ('scoring_impact', 6), ('eff_times_minutes', 6), 
        ('scoring_volume', 6)
    ]

    for col, n_bins in bining_cols:
        # Standard binning
        std_bin_col = f'{col}_binning_standard'
        std_bins = bining.standard_binning(
            df[col],
            n_bins=n_bins,
            labels=list(range(1, n_bins + 1))
        )

        # Quantile binning
        q_bin_col = f'{col}_binning_quantile'
        q_bins = bining.quantile_binning(
            df[col],
            n_bins=n_bins,
            labels=list(range(1, n_bins + 1))
        )
        datasets['ds3'][std_bin_col] = std_bins
        datasets['ds4'][q_bin_col] = q_bins

    # ------------------------------------------------------------------
    # One-hot encoding
    # ------------------------------------------------------------------
    ds3_ohe_cols = list(set(datasets['ds3'].columns) - {'target', 'player_id'})
    ds4_ohe_cols = list(set(datasets['ds4'].columns) - {'target', 'player_id'})

    datasets['ds5'] = pd.concat(
        [datasets['ds5'], encoders.one_hot_encoding(datasets['ds3'][ds3_ohe_cols])],
        axis=1
    )

    datasets['ds6'] = pd.concat(
        [datasets['ds6'], encoders.one_hot_encoding(datasets['ds4'][ds4_ohe_cols])],
        axis=1
    )

    # ------------------------------------------------------------------
    # Sort & save
    # ------------------------------------------------------------------
    for name, ds in datasets.items():
        ds.sort_values(by='player_id', inplace=True)
        ds.to_csv(f'{df_version}_{name}.csv', index=True)
