import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils.storage import path_validate


def correlation_and_heatmap(
    df: pd.DataFrame,
    figures_path: str,
    figure_name: str,
    ds_name: str,
    method: str = "pearson",
    annot: bool = False,
) -> None:
    """
    Compute correlation matrix and save heatmap.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing continuous variables only.
    figures_path : str
        Base directory for figures.
    figure_name : str
        Output file name (without extension).
    method : str
        Correlation method: 'pearson', 'spearman', or 'kendall'.
    annot : bool
        Whether to annotate correlation values.
    """
    corr = df.corr(method=method)

    plt.figure(figsize=(18, 16))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        annot=annot,
        fmt=".2f",
        square=True,
        cbar_kws={"shrink": 0.8},
    )
    plt.title(f"Correlation Heatmap ({method.capitalize()})")
    plt.tight_layout()

    file_path = f"{figures_path}correlations/{ds_name}/{figure_name}_heatmap.png"
    path_validate(file_path)
    plt.savefig(file_path, dpi=300)
    plt.close()


def continuous_features_correlation_analysis(
    datasets: dict,
    figures_path: str,
) -> None:
    """
    Generate correlation heatmaps for continuous variables
    across multiple datasets.

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]
        Dictionary of datasets.
    figures_path : str
        Base directory for figures.
    """
    prefixes = (
        "minutes_played",
        "fg_pct",
        "three_pct",
        "ft_pct",
        "eff_per_point",
        "eff_per_min",
        "points_per_min",
        "scoring_impact",
        "eff_times_minutes",
        "scoring_volume",
    )

    for ds_name, df in datasets.items():
        continuous_cols = [
            col
            for col in df.columns
            if col.startswith(prefixes)
        ]

        if continuous_cols:
            correlation_and_heatmap(
                df=df[continuous_cols],
                figures_path=figures_path,
                ds_name=ds_name,
                figure_name=f"{ds_name}_continuous",
            )

        numeric_df = df.select_dtypes(include="number")
        if not numeric_df.empty:
            correlation_and_heatmap(
                df=numeric_df,
                figures_path=figures_path,
                ds_name=ds_name,
                figure_name=f"{ds_name}_all_numeric",
            )
