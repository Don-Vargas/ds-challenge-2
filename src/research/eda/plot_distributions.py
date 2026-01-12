import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.storage import path_validate

def plot_continuous(series, base_path, bins=30):
    stats_dict = compute_stats(series)

    fig, axes = plt.subplots(
        nrows=2,
        figsize=(8, 6),
        gridspec_kw={'height_ratios': [3, 1]}
    )

    # Histogram + KDE
    sns.histplot(
        series.dropna(),
        bins=bins,
        kde=True,
        ax=axes[0]
    )
    axes[0].set_title(f'Distribution of {series.name}')

    # Boxplot
    sns.boxplot(
        x=series.dropna(),
        ax=axes[1],
        orient='h'
    )

    # Stats box
    text = (
        f"Mean: {stats_dict['mean']:.2f}\n"
        f"Median: {stats_dict['median']:.2f}\n"
        f"Skewness: {stats_dict['skew']:.2f}\n"
        f"Kurtosis: {stats_dict['kurtosis']:.2f}\n"
        f"KS p-value: {stats_dict['ks_p']:.4f}\n"
        f"Outliers: {stats_dict['outliers']} "
        f"({stats_dict['outlier_pct']:.1f}%)"
    )

    axes[0].text(
        0.98, 0.95,
        text,
        transform=axes[0].transAxes,
        ha='right',
        va='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    plt.tight_layout()

    file_path = f"{base_path}numerical/continuous/{series.name}_distribution.png"
    path_validate(file_path)
    plt.savefig(file_path, dpi=300)
    plt.close()

def plot_discrete(series, base_path):
    stats_dict = compute_stats(series)

    fig, axes = plt.subplots(
        nrows=2,
        figsize=(8, 6),
        gridspec_kw={'height_ratios': [3, 1]}
    )

    # Count plot
    sns.countplot(
        x=series,
        order=series.value_counts().index,
        ax=axes[0]
    )
    axes[0].set_title(f'Distribution of {series.name}')
    axes[0].tick_params(axis='x', rotation=45)

    # Boxplot
    sns.boxplot(
        x=series.dropna(),
        ax=axes[1],
        orient='h'
    )

    text = (
        f"Mean: {stats_dict['mean']:.2f}\n"
        f"Median: {stats_dict['median']:.2f}\n"
        f"Skewness: {stats_dict['skew']:.2f}\n"
        f"Kurtosis: {stats_dict['kurtosis']:.2f}\n"
        f"KS p-value: {stats_dict['ks_p']:.4f}\n"
        f"Outliers: {stats_dict['outliers']} "
        f"({stats_dict['outlier_pct']:.1f}%)"
    )

    axes[0].text(
        0.98, 0.95,
        text,
        transform=axes[0].transAxes,
        ha='right',
        va='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    plt.tight_layout()

    file_path = f"{base_path}numerical/discrete/{series.name}_distribution.png"
    path_validate(file_path)
    plt.savefig(file_path, dpi=300)
    plt.close()


def compute_stats(series):
    s = series.dropna()

    # IQR outliers
    q1, q3 = np.percentile(s, [25, 75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = ((s < lower) | (s > upper)).sum()

    # KS normality test
    if s.std() > 0:
        ks_stat, ks_p = stats.kstest(
            (s - s.mean()) / s.std(),
            'norm'
        )
    else:
        ks_stat, ks_p = np.nan, np.nan

    return {
        "mean": s.mean(),
        "median": s.median(),
        "skew": stats.skew(s),
        "kurtosis": stats.kurtosis(s),
        "ks_p": ks_p,
        "outliers": outliers,
        "outlier_pct": outliers / len(s) * 100
    }

def plot_categorical(series, base_path):
    plt.figure(figsize=(8, 4))

    sns.countplot(
        x=series.dropna(),
        order=series.value_counts().index
    )

    plt.title(f"Distribution of {series.name}")
    plt.xlabel(series.name)
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    file_path = f"{base_path}categorical/{series.name}_distribution.png"
    path_validate(file_path)
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()

def all_pairplots(df, output_path):
    sns.pairplot(
        data=df,
        diag_kind="kde",
        corner=True
    )

    file_path = f"{output_path}pair_plots/all_pair_plots.png"
    path_validate(file_path)

    plt.tight_layout()
    plt.savefig(file_path, dpi=300)
    plt.close()

def generate_plots(data, output_path):
    df = pd.read_csv(data, index_col='row_id')

    # Categorical features (nominal)
    categorical_nominal_columns = [
        'position',
        'team',
        'opponent',
        'game_location'
    ]

    # Categorical features (ordinal)
    categorical_ordinal_columns = [
        'rest_days'
    ]

    # Numerical features – continuous
    numerical_continuous_columns = [
        'minutes_played',
        'fg_pct',
        'three_pct',
        'ft_pct'
    ]

    # Numerical features – discrete (counts / integer measures)
    numerical_discrete_columns = [
        'age',
        'plus_minus',
        'efficiency',
        'points',
        'rebounds',
        'assists',
        'steals',
        'blocks',
        'turnovers'
    ]

    # Target variable (binary categorical)
    target_column = ['target']

    categorical_columns = categorical_nominal_columns + categorical_ordinal_columns + target_column

    for col in categorical_columns:
        plot_categorical(df[col], output_path)

    for col in numerical_continuous_columns:
        plot_continuous(df[col], output_path)

    for col in numerical_discrete_columns:
        plot_continuous(df[col], output_path)

    all_pairplots(df, output_path)

