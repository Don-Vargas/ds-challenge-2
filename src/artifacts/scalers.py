from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer

# Min-Max Scaling
def minimaxation(column_df):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(column_df)
    return scaled_data

# Standardization (Z-score)
def standardization(column_df):
    std_scaler = StandardScaler()
    standardized_data = std_scaler.fit_transform(column_df)
    return standardized_data

# L1 Normalization (Manhattan norm)
def l1_normalization(column_df):
    normalizer = Normalizer(norm='l1')
    normalized_data = normalizer.fit_transform(column_df)
    return normalized_data

# L2 Normalization (Euclidean norm)
def l2_normalization(column_df):
    normalizer = Normalizer(norm='l2')
    normalized_data = normalizer.fit_transform(column_df)
    return normalized_data
