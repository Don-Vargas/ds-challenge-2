import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import entropy

def common_summary(series: pd.Series) -> dict:
    s = series.dropna()
    dtype = series.dtype
    value_counts = s.value_counts()

    min_val = s.min() if np.issubdtype(dtype, np.number) else np.nan
    max_val = s.max() if np.issubdtype(dtype, np.number) else np.nan
    mode_val = value_counts.index[0] if not value_counts.empty else np.nan

    return {
        'Column': series.name,
        'Type': dtype,
        'Unique Values': series.nunique(dropna=True),
        'Missing': series.isnull().sum(),
        '% Missing': series.isnull().mean(),
        'Keep': True,
        'Min': min_val,
        'Max': max_val,
        'Mode': mode_val
    }

def numeric_details(series: pd.Series) -> dict:
    s = series.dropna()
    mean = s.mean()
    std = s.std()
    skewness = stats.skew(s)
    kurt = stats.kurtosis(s)

    # Normality tests
    p_shapiro = stats.shapiro(s).pvalue if len(s) <= 5000 else np.nan
    normal_shapiro = int(p_shapiro > 0.05) if not np.isnan(p_shapiro) else np.nan

    if std > 0:
        _, p_ks = stats.kstest(s, 'norm', args=(mean, std))
        normal_ks = int(p_ks > 0.05)
    else:
        p_ks, normal_ks = np.nan, np.nan

    return {
        'Mean': mean,
        'Median': s.median(),
        'Variance': s.var(),
        'Std Dev': std,
        'Skewness': skewness,
        'Kurtosis': kurt,
        'Shapiro p-value': p_shapiro,
        'Normal (Shapiro)': normal_shapiro,
        'KS p-value': p_ks,
        'Normal (KS)': normal_ks
    }

def categorical_details(series: pd.Series) -> dict:
    s = series.dropna()
    value_counts = s.value_counts()
    probs = value_counts / value_counts.sum()

    return {
        'Mode Count': value_counts.iloc[0],
        '% Mode': probs.iloc[0],
        'Entropy': entropy(probs) if len(probs) > 1 else 0,
        'High Cardinality': int(s.nunique() / len(s) > 0.5),
        'Binary': int(s.nunique() == 2),
        'Rare Categories': int((probs < 0.01).sum())
    }

def explore_dataset(df_path: str, simple_report: str, detailed_report: str, include_categorical: bool = False):
    df = pd.read_csv(df_path)

    simple_list = []
    detailed_list = []

    for column in df.columns:
        series = df[column]
        dtype = series.dtype

        # Common summary
        common = common_summary(series)
        simple_list.append(common)

        # Numeric or categorical details
        if np.issubdtype(dtype, np.number):
            details = numeric_details(series)
            detailed_list.append({**common, **details})
        elif include_categorical:
            details = categorical_details(series)
            detailed_list.append({**common, **details})

    # Save reports
    pd.DataFrame(simple_list).to_csv(simple_report, index=False)
    pd.DataFrame(detailed_list).to_csv(detailed_report, index=False)
