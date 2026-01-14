from collections import defaultdict
from src.artifacts import encoders, scalers, bining
from src.artifacts.dimentionality_reduction import pca


#--------------------------------
def apply_raw_numeric(df_temp, cols):
    new_cols = df_temp[cols].copy()
    return new_cols

#--------------------------------
def apply_pca(df, role='train', pca_config=None):
    if role == 'train':
        pca_results = pca.pca_by_variance(df)

    else:
        pca_results = pca.pca_transform(df, pca_config)

    return pca_results['X_reduced'], pca_results['pca_config']

#--------------------------------
def apply_binning(dataset_name, datasets, cols, mode, role='train', bin_config=defaultdict(dict)):
    if role == 'train':
        bin_config_list = []
        for col, n_bins in cols:
            if mode == "standard":
                binning_values, col_config = bining.standard_binning(datasets[dataset_name][col], n_bins, labels=range(1, n_bins + 1))
                bin_config_list.append({f"{col}_bin": col_config})
                datasets[dataset_name][f"{col}_binning_standard"] = binning_values
                datasets[dataset_name].drop(columns=[col], inplace=True, errors="ignore")

            elif mode == "quantile":
                binning_values, col_config = bining.quantile_binning(
                    datasets[dataset_name][col], n_bins, labels=range(1, n_bins + 1))
                bin_config_list.append({f"{col}_bin": col_config})
                datasets[dataset_name][f"{col}_binning_quantile"] = binning_values
                datasets[dataset_name].drop(columns=[col], inplace=True, errors="ignore")
        bin_config = {mode : bin_config_list}
    else:
        for col, _ in cols:
            binning_values, _ = bining.apply_back_binning(
                datasets[dataset_name][col], bin_config)
            suffix = "_binning_standard" if mode == "standard" else "_binning_quantile"
            datasets[dataset_name][f"{col}{suffix}"] = binning_values
            datasets[dataset_name].drop(columns=[col], inplace=True, errors="ignore")
        bin_config = None

    return datasets, bin_config

#--------------------------------
def apply_scaling(dataset_name, datasets, cols, scale_type, role = 'train', scaling_config=defaultdict(dict)):
    if role == 'train':
        scaling_config_list = []
        for col in cols:
            if scale_type == "standard": 
                scaled_values, col_config = scalers.standardization(datasets[dataset_name][col])
                scaling_config_list.append({f"{col}_scaler": col_config})
                datasets[dataset_name][f"{col}_standard"] = scaled_values
                datasets[dataset_name].drop(columns=[col], inplace=True, errors="ignore")
            elif scale_type == "minmax":
                scaled_values, col_config = scalers.minimaxation(datasets[dataset_name][col])
                scaling_config_list.append({f"{col}_scaler": col_config})
                datasets[dataset_name][f"{col}_minmax"] = scaled_values
                datasets[dataset_name].drop(columns=[col], inplace=True, errors="ignore")
            scaling_config = scaling_config_list
    else:
        for col in cols:
            scaled_values = scalers.apply_scaling_back(
                datasets[dataset_name][col], scaling_config)
            datasets[dataset_name][f"{col}_{scale_type}"] = scaled_values
            datasets[dataset_name].drop(columns=[col], inplace=True, errors="ignore")
        scaling_config = None
    return datasets, scaling_config

#--------------------------------
def apply_frequency_encoding(dataset_name, datasets, cols, role='train', freq_config=defaultdict(dict)):
    # Determine which datasets to update
    if role == 'train':
        freq_config_list = []
        for col in cols:
            freq_values, col_config = encoders.frequency_encoding(datasets[dataset_name][col])
            freq_config_list.append({f"{col}_freq": col_config})
            datasets[dataset_name][f"{col}_freq"] = freq_values
            datasets[dataset_name].drop(columns=[col], inplace=True, errors="ignore")
        freq_config = {'frequency_encoding': freq_config_list}
    else:
        for col in cols:
            freq_values, _ = encoders.apply_back_frequency_encoding(
                datasets[dataset_name][col], freq_config)
            datasets[dataset_name][f"{col}_freq"] = freq_values
            datasets[dataset_name].drop(columns=[col], inplace=True, errors="ignore")
        freq_config = None

    return datasets, freq_config

#--------------------------------
def apply_one_hot(df, role = 'train', one_hot_config=defaultdict(dict)):
    if role == 'train':
        encoded_df, one_hot_config = encoders.one_hot_encoding(df)
    else:
        encoded_df = encoders.apply_back_one_hot_encoding(df, one_hot_config)
    return encoded_df, one_hot_config


#--------------------------------
def feature_engineering_pipeline(datasets, ds_keys, processing_configs=None, role = 'train'):
    df_temp = datasets['ds1'].copy()
    if role == 'train':
        processing_configs = defaultdict(dict)

    for ds, cfg in ds_keys.items():
        '''
        # Raw Numeric
        if "raw_numeric" in cfg:
            print(ds)
            datasets[ds] = pd.concat([datasets[ds], apply_raw_numeric(df_temp, cfg["raw_numeric"])], axis=1)
        '''

        # Frequency Encoding
        if "frequency_encoding" in cfg:
            datasets, freq_config = apply_frequency_encoding(
                ds,
                datasets,
                cfg["frequency_encoding"],
                role=role,
                freq_config=processing_configs
            )
            processing_configs[ds]['frequency_encoding'] = freq_config

        # Scaling
        if "scaling" in cfg:
            for scale_type, cols in cfg["scaling"].items():
                datasets, scaling_config = apply_scaling(ds, datasets, cols, scale_type, role = role)
            processing_configs[ds][scale_type] = scaling_config

        # Binning
        if "binning" in cfg:
            for mode, cols in cfg["binning"].items():
                datasets, bin_config = apply_binning(
                    ds, 
                    datasets, 
                    cols, 
                    mode,
                    role = role)

                processing_configs[ds][f'bin_{mode}'] = bin_config

    # One-Hot Encoding
    one_hot_mapping = {ds: cfg["one_hot_from"] for ds, cfg in ds_keys.items() if "one_hot_from" in cfg}
    if one_hot_mapping:
        for target_ds, source_ds in one_hot_mapping.items():
            datasets[target_ds], one_hot_config = apply_one_hot(datasets[source_ds], role =role)
            processing_configs[target_ds]['one_hot'] = one_hot_config

    # PCA
    pca_mapping = {ds: cfg["pca_from"] for ds, cfg in ds_keys.items() if "pca_from" in cfg}
    if one_hot_mapping:
        for target_ds, source_ds in pca_mapping.items():
            datasets[target_ds], pca_config = apply_pca(datasets[source_ds], role=role)
            processing_configs[target_ds]['pca'] = pca_config

    return datasets, processing_configs
