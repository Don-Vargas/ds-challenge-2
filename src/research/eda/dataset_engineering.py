import pandas as pd
from src.artifacts import encoders, scalers, bining
from src.artifacts.dimentionality_reduction import pca


#--------------------------------
def apply_raw_numeric(datasets, cols, df_temp):
    new_cols = df_temp[cols].copy()
    return new_cols



#--------------------------------
def apply_pca(df, role='train', pca_config=None):
    if role == 'train':
        X_reduced, n_components_, explained_variance_ratio_, pca_config = pca.pca_by_variance(df)
        return X_reduced, n_components_, explained_variance_ratio_, pca_config
    else:
        X_reduced = pca.pca_transform(df, pca_config)
        return X_reduced, None, None, pca_config



#--------------------------------
def apply_one_hot(datasets, one_hot_mapping):
    for target_ds, source_ds in one_hot_mapping.items():
        datasets[target_ds] = pd.concat(
            [datasets[target_ds], encoders.one_hot_encoding(datasets[source_ds])],
            axis=1
        )
    return datasets

#--------------------------------
def apply_binning(df, datasets, cols, mode, ds, role='train', bin_config=None):
    if role == 'train':
        for col, n_bins in cols:
            if mode == "standard":
                binning_cols, bin_config = bining.standard_binning(
                    df[col], n_bins, labels=range(1, n_bins + 1)
                )
                datasets[ds][f"{col}_binning_standard"] = binning_cols
            elif mode == "quantile":
                binning_cols, bin_config = bining.quantile_binning(
                    df[col], n_bins, labels=range(1, n_bins + 1)
                )
                datasets[ds][f"{col}_binning_quantile"] = binning_cols
    else:
        for col, _ in cols:
            binning_cols, bin_config = bining.apply_back_binning(df[col], bin_config)
            # Keep the mode consistent when applying
            suffix = "_binning_standard" if mode == "standard" else "_binning_quantile"
            datasets[ds][f"{col}{suffix}"] = binning_cols

    return datasets, bin_config
#--------------------------------
def apply_frequency_encoding(dataset_name, datasets, cols, role='train', freq_config=None):
    # Determine which datasets to update
    if role == 'train':
        for col in cols:
            freq_values, col_config = encoders.frequency_encoding(datasets[dataset_name][col])
            freq_config.setdefault('frequency_encoding', {}).setdefault(dataset_name, {})[f"{col}_freq"] = col_config
            datasets[dataset_name][f"{col}_freq"] = freq_values
            del datasets[dataset_name][col]
    else:
        for col in cols:
            freq_values, freq_config = encoders.apply_back_frequency_encoding(
                datasets[dataset_name][col], freq_config)
            datasets[dataset_name][f"{col}_freq"] = freq_values
            del datasets[dataset_name][col]

    return datasets, freq_config

#--------------------------------
def apply_scaling(dataset_name, datasets, cols, scale_type):
    for col in cols:
        if scale_type == "standard":
            datasets[dataset_name][f"{col}_standard"] = scalers.standardization(datasets[dataset_name][col])
            del datasets[dataset_name][col]
        elif scale_type == "minmax":
            datasets[dataset_name][f"{col}_minmax"] = scalers.minimaxation(datasets[dataset_name][col])
            del datasets[dataset_name][col]
    return datasets

#--------------------------------
def feature_engineering_pipeline(datasets, ds_keys, processing_configs=None, role = 'train'):
    df_temp = datasets['ds1'].copy()
    if processing_configs is None:
        processing_configs = {}

    for ds, cfg in ds_keys.items():
        # Raw Numeric
        if "raw_numeric" in cfg:
            datasets[ds] = pd.concat([datasets[ds], apply_raw_numeric(datasets, cfg["raw_numeric"], df_temp)], axis=1)

        # Frequency Encoding
        if "frequency_encoding" in cfg:
            datasets, freq_config = apply_frequency_encoding(
                ds,
                datasets,
                cfg["frequency_encoding"],
                role=role,
                freq_config=processing_configs
            )

        # Scaling
        if "scaling" in cfg:
            for scale_type, cols in cfg["scaling"].items():
                datasets = apply_scaling(ds, datasets, cols, scale_type)

        # Binning
        if "binning" in cfg:
            for mode, cols in cfg["binning"].items():
                datasets, bin_config = apply_binning(datasets[ds], datasets, cols, mode, ds, role = role)

        processing_configs[ds] = {"freq_config": freq_config, 
                                  "bin_config": bin_config}


    print(datasets['ds1'].columns)#['turnovers'])
    return None, None
'''
    # One-Hot Encoding
    one_hot_mapping = {ds: cfg["one_hot_from"] for ds, cfg in ds_keys.items() if "one_hot_from" in cfg}
    if one_hot_mapping:
        datasets = apply_one_hot(datasets, one_hot_mapping)

    # PCA
    pca_mapping = {ds: cfg["pca_from"] for ds, cfg in ds_keys.items() if "pca_from" in cfg}
    for target_ds, source_ds in pca_mapping.items():
        X_reduced, _, _, pca_config = apply_pca(
            datasets[source_ds],
            role=role,
            pca_config=processing_configs.get(source_ds, {}).get("pca_config")
        )
        datasets[target_ds] = pd.DataFrame(
            X_reduced,
            index=datasets[source_ds].index,
            columns=[f"PC{i+1}" for i in range(X_reduced.shape[1])]
        )
        processing_configs[target_ds] = {"pca_config": pca_config}


    return datasets, processing_configs
'''