from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
import numpy as np

# Min-Max Scaling
def minimaxation(column_df):
    scaler = MinMaxScaler()
    # Reshape the column to be 2D
    scaled_data = scaler.fit_transform(column_df.values.reshape(-1, 1))
    return scaled_data

# Standardization (Z-score)
def standardization(column_df):
    std_scaler = StandardScaler()
    # Reshape the column to be 2D
    standardized_data = std_scaler.fit_transform(column_df.values.reshape(-1, 1))
    return standardized_data
