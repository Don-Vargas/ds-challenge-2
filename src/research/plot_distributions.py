import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from src.utils.storage import path_validate

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # change to DEBUG for more detail

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(levelname)s] %(name)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)



def compute_stats(series):
    """Compute descriptive, robust, and normality statistics."""
    s = series.dropna()
    n = len(s)

    if n == 0:
        logger.warning(f"No data available for series '{series.name}'")
        return {
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "var": np.nan,
            "min": np.nan,
            "max": np.nan,
            "iqr": np.nan,
            "mad": np.nan,
            "skew": np.nan,
            "kurtosis": np.nan,
            "ks_p": np.nan,
            "jb_p": np.nan,
            "shapiro_p": np.nan,
            "outliers": 0,
            "outlier_pct": 0.0,
        }

    logger.debug(f"Computing stats for '{series.name}' (n={n})")

    q1, q3 = np.percentile(s, [25, 75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = ((s < lower) | (s > upper)).sum()

    std = s.std()
    var = s.var()
    mad = stats.median_abs_deviation(s, scale="normal")

    # Normality tests
    if std > 0:
        _, ks_p = stats.kstest((s - s.mean()) / std, "norm")
        _, jb_p = stats.jarque_bera(s)
    else:
        ks_p = np.nan
        jb_p = np.nan

    if 3 <= n <= 5000:
        _, shapiro_p = stats.shapiro(s)
    else:
        shapiro_p = np.nan

    return {
        "mean": s.mean(),
        "median": s.median(),
        "std": std,
        "var": var,
        "min": s.min(),
        "max": s.max(),
        "iqr": iqr,
        "mad": mad,
        "skew": stats.skew(s),
        "kurtosis": stats.kurtosis(s),
        "ks_p": ks_p,
        "jb_p": jb_p,
        "shapiro_p": shapiro_p,
        "outliers": outliers,
        "outlier_pct": (outliers / n) * 100,
    }


def plot_continuous(series, base_path, bins=30):
    """Plot histogram, KDE, boxplot, and extended statistics."""
    logger.info(f"Generating continuous plot for '{series.name}'")

    s = series.dropna()
    stats_dict = compute_stats(series)

    fig, axes = plt.subplots(
        nrows=2,
        figsize=(8, 6),
        gridspec_kw={"height_ratios": [3, 1]},
    )

    sns.histplot(s, bins=bins, kde=True, ax=axes[0])
    axes[0].set_title(f"Distribution of {series.name}")

    sns.boxplot(x=s, ax=axes[1], orient="h")

    stats_text = (
        f"Mean: {stats_dict['mean']:.2f}\n"
        f"Median: {stats_dict['median']:.2f}\n"
        f"Std: {stats_dict['std']:.2f}\n"
        f"IQR: {stats_dict['iqr']:.2f}\n"
        f"Skew: {stats_dict['skew']:.2f}\n"
        f"Kurtosis: {stats_dict['kurtosis']:.2f}\n"
        f"JB p-value: {stats_dict['jb_p']:.4f}\n"
        f"Shapiro p: {stats_dict['shapiro_p']:.4f}\n"
        f"Outliers: {stats_dict['outliers']} "
        f"({stats_dict['outlier_pct']:.1f}%)"
    )

    axes[0].text(
        0.98,
        0.95,
        stats_text,
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )

    plt.tight_layout()

    file_path = f"{base_path}numerical/{series.name}_distribution.png"
    path_validate(file_path)
    plt.savefig(file_path, dpi=300)
    plt.close(fig)

    logger.info(f"Saved plot : {file_path}")


def plot_categorical(series, base_path):
    """Plot full categorical distribution (no cardinality cutoff)."""
    logger.info(f"Generating categorical plot for '{series.name}'")

    s = series.dropna()
    order = s.value_counts().index

    plt.figure(figsize=(8, 4))
    sns.countplot(x=s, order=order)

    plt.title(f"Distribution of {series.name}")
    plt.xlabel(series.name)
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    file_path = f"{base_path}categorical/{series.name}_distribution.png"
    path_validate(file_path)
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved plot: {file_path}")


def all_pairplots(df, output_path):
    """Generate pair plots for all numeric variables."""
    logger.info("Generating pair plots (this may take a while)")

    sns.pairplot(df, diag_kind="kde", corner=True)

    file_path = f"{output_path}pair_plots/all_pair_plots.png"
    path_validate(file_path)

    plt.tight_layout()
    plt.savefig(file_path, dpi=300)
    plt.close()
    logger.info(f"Saved pair plots : {file_path}")


def generate_plots(df, output_path):
    """Run full EDA plot generation."""
    logger.info("Starting EDA plot generation")

    categorical_nominal_columns = [
        "position",
        "team",
        "opponent",
        "game_location",
    ]

    categorical_ordinal_columns = ["rest_days"]

    numerical_continuous_columns = [
        "minutes_played",
        "fg_pct",
        "three_pct",
        "ft_pct",
    ]

    numerical_discrete_columns = [
        "age",
        "plus_minus",
        "efficiency",
        "points",
        "rebounds",
        "assists",
        "steals",
        "blocks",
        "turnovers",
    ]

    engineered_numerical_columns = [
        "scoring_impact",
        "eff_per_point",
        "scoring_volume",
        "eff_per_min",
        "eff_times_minutes",
    ]

    target_column = ["target"]

    categorical_columns = (
        categorical_nominal_columns
        + categorical_ordinal_columns
        + target_column
    )

    numerical_columns = (
        numerical_continuous_columns
        + numerical_discrete_columns
        + engineered_numerical_columns
    )

    for col in categorical_columns:
        logger.debug(f"Processing categorical column: {col}")
        plot_categorical(df[col], output_path)

    for col in numerical_columns:
        logger.debug(f"Processing numerical column: {col}")
        plot_continuous(df[col], output_path)

    all_pairplots(df, output_path)

    logger.info("EDA plot generation completed")
