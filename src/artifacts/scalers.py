import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Min-Max Scaling
def minimaxation(column_df):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(column_df.values.reshape(-1, 1))
    return scaled_data, scaler

# Standardization (Z-score)
def standardization(column_df):
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(column_df.values.reshape(-1, 1))
    return standardized_data, scaler

def apply_scaling_back(column_df, fitted_scalers):
    scaler = fitted_scalers
    original_data = scaler.inverse_transform(column_df.values.reshape(-1, 1))
    return pd.Series(original_data.flatten(), index=column_df.index)