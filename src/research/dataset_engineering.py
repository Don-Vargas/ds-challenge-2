from collections import defaultdict
from collections import ChainMap
from src.artifacts.preprocessing import encoders, scalers, bining
from src.artifacts.dimentionality_reduction import pca


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
        config = freq_config[dataset_name]['frequency_encoding']['frequency_encoding']
        dict_config = ChainMap(*config)

        df = datasets[dataset_name]
        apply_enc = encoders.apply_back_frequency_encoding

        for col in cols:
            freq_values, _ = apply_enc(df[col], dict_config[f"{col}_freq"])
            df[f"{col}_freq"] = freq_values

        df.drop(columns=cols, inplace=True, errors="ignore")
        freq_config = None

    return datasets, freq_config

#--------------------------------
def apply_raw_numeric(df_temp, cols):
    new_cols = df_temp[cols].copy()
    return new_cols

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
        config = bin_config[dataset_name]
        binning_type_key = next(k for k in config if k != 'frequency_encoding')
        bins_list = config[binning_type_key][mode]
        dict_config = dict(ChainMap(*bins_list))
        df = datasets[dataset_name]
        apply_enc = bining.apply_back_binning
        suffix = "_binning_standard" if mode == "standard" else "_binning_quantile"
        for col, _ in cols:
            col_config = dict_config.get(f"{col}_bin")
            if col_config is not None:
                df[f"{col}{suffix}"], _ = apply_enc(df[col], col_config)
                df.drop(columns=[col], inplace=True, errors="ignore")
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
        config = scaling_config[dataset_name]
        scaler_type_key = next(k for k in config if k != 'frequency_encoding')
        dict_config = ChainMap(*config[scaler_type_key])
        df = datasets[dataset_name]
        apply_enc = scalers.apply_scaling_back

        for col in cols:
            scaler = dict_config.get(f"{col}_scaler")
            df[f"{col}_{scaler_type_key}"] = apply_enc(df[col], scaler)

        df.drop(columns=cols, inplace=True, errors="ignore")
        scaling_config = None
    return datasets, scaling_config

#--------------------------------
def apply_one_hot(target_ds, df, role = 'train', one_hot_config=defaultdict(dict)):
    if role == 'train':
        encoded_df, one_hot_config = encoders.one_hot_encoding(df)
    else:
        config = one_hot_config[target_ds]['one_hot']
        encoded_df = encoders.apply_back_one_hot_encoding(df, config)
    return encoded_df, one_hot_config

#--------------------------------
def apply_pca(target_ds, df, role='train', pca_config=None):
    if role == 'train':
        pca_results = pca.pca_by_variance(df)

    else:
        config = pca_config[target_ds]#['pca']
        if df.isna().any().any():  
            print("Rows containing NaN values:")
            df = df.dropna()
        pca_results = pca.pca_transform(df, config)

    return pca_results['X_reduced'], pca_results['pca_config']

#--------------------------------
def feature_engineering_pipeline(datasets, ds_keys, processing_configs=None, role = 'train'):
    '''
    df_temp = datasets['ds1'].copy()
    '''
    if role == 'train':
        processing_configs = defaultdict(dict)
        freq_config = defaultdict(dict)

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
                datasets, scaling_config = apply_scaling(
                    ds, 
                    datasets, 
                    cols, 
                    scale_type, 
                    role = role, 
                    scaling_config=processing_configs)
            processing_configs[ds][scale_type] = scaling_config

        # Binning
        if "binning" in cfg:
            for mode, cols in cfg["binning"].items():
                datasets, bin_config = apply_binning(
                    ds, 
                    datasets, 
                    cols, 
                    mode,
                    role = role,
                    bin_config = processing_configs)

                processing_configs[ds][f'bin_{mode}'] = bin_config

    # One-Hot Encoding
    one_hot_mapping = {ds: cfg["one_hot_from"] for ds, cfg in ds_keys.items() if "one_hot_from" in cfg}
    if one_hot_mapping:
        for target_ds, source_ds in one_hot_mapping.items():
            datasets[target_ds], one_hot_config = apply_one_hot(target_ds, datasets[source_ds], role =role, one_hot_config = processing_configs)
            processing_configs[target_ds]['one_hot'] = one_hot_config

    # PCA
    pca_mapping = {ds: cfg["pca_from"] for ds, cfg in ds_keys.items() if "pca_from" in cfg}
    if one_hot_mapping:
        for target_ds, source_ds in pca_mapping.items():
            datasets[target_ds], pca_config = apply_pca(target_ds, datasets[source_ds], role=role, pca_config=processing_configs)
            processing_configs[target_ds]['pca'] = pca_config

    return datasets, processing_configs
