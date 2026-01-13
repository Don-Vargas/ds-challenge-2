import pandas as pd

from src.utils.storage import path_validate
from src.artifacts.dimentionality_reduction.pca import pca_by_variance

# Function to select columns with flexible matching
def select_columns(df, features):
    selected_columns = []

    # Case 1: tuple features (prefix + suffix)
    if features and isinstance(features[0], tuple):
        for prefix, suffix in features:
            if suffix:
                # Match columns that start with prefix_ and end with _suffix
                matched = [col for col in df.columns if col.startswith(f"{prefix}_") and col.endswith(f"_{suffix}")]
            else:
                # Match columns that start with prefix
                matched = [col for col in df.columns if col.startswith(prefix)]
            selected_columns.extend(matched)

    # Case 2: simple list of feature prefixes
    else:
        for prefix in features:
            # Match columns that contain the prefix anywhere (robust to minor typos/suffixes)
            matched = [col for col in df.columns if prefix in col]
            selected_columns.extend(matched)

    # Remove duplicates while preserving order
    return list(dict.fromkeys(selected_columns))


def dataset_building(datasets_path):
    ds_path = f'{datasets_path}/training_dataset'
    path_validate(ds_path)

    # Apply PCA for datasets _ds1 to _ds4
    for ds in ['_ds1', '_ds2', '_ds3', '_ds4']:
        df = pd.read_csv(f'{datasets_path}{ds}.csv', index_col='row_id')

        column_drop = ['target', 'player_id','original_position','original_team', 
                       'original_opponent', 'original_game_location', 
                       'original_rest_days', 'original_high_usage_scorer', 
                       'original_high_eff_min','original_high_eff_scorer']
        existing_cols = [c for c in column_drop if c in df.columns]
        X = df.drop(columns=existing_cols)
        X.columns = X.columns.astype(str)
        y = df['target']
        z = df['player_id']

        X_reduced, n_components, var_ratio = pca_by_variance(X)
        X_reduced_df = pd.DataFrame(
            X_reduced, 
            index=df.index, 
            columns=[f'PC{i+1}' for i in range(X_reduced.shape[1])]
        )

        df_reduced = pd.concat([X_reduced_df, y, z], axis=1)
        df_reduced.to_csv(f'{ds_path}/{ds}_pca.csv', index=True)

    # Define feature selections for different datasets
    feature_configs = {
        ('_ds1', '_ds2'): ['eff_per_min', 'efficiency', 'scoring_impact',
                            'eff_per_point', 'eff_times_minutes', 'plus_minus',
                            'turnovers', 'scoring_volume', 'target', 'player_id'],

        ('_ds3', '_ds4'): ['efficiency', 'scoring_impact', 'eff_per_point', 
                            'eff_times_minutes', 'plus_minus', 'turnovers', 
                            'scoring_volume', 'opponent', 'team', 'target', 'player_id'],

        ('_ds5', '_ds6'): [('eff_per_min', '6'), ('efficiency', '10'), 
                            ('scoring_impact', '2'), ('scoring_impact', '3'), 
                            ('eff_per_point', '2'), ('eff_times_minutes', '4'), 
                            ('eff_times_minutes', '2'), ('eff_times_minutes', '3'), 
                            ('high_eff_min', '0.7963888888888889'), ('high_eff_scorer', '0.5361111111111111'), 
                            ('assists','4'), ('target', ''), ('player_id','')]
    }

    # Process all datasets
    for datasets, features in feature_configs.items():
        for ds in datasets:
            df = pd.read_csv(f'{datasets_path}{ds}.csv', index_col='row_id')

            selected_columns = select_columns(df, features)
            
            # Reduce dataframe
            df_reduced = df[selected_columns]
            # Save reduced dataframe
            df_reduced.to_csv(f'{ds_path}/{ds}_important_f.csv', index=True)
