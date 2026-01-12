import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.utils.storage import path_validate


    # Assuming your dataframe is 'df' and contains only continuous features
def correlation_and_heatmap(df, figures_path, figure_name):
    correlation_matrix = df.corr(method='pearson')  # or 'spearman', 'kendall'
    plt.figure(figsize=(18, 16))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title("Correlation Heatmap")
    
    
    file_path = f"{figures_path}correlations/{figure_name}_heatmap.png"
    path_validate(file_path)
    plt.savefig(file_path, dpi=300)
    plt.close()

# Main function to test and visualize categorical and numeric data
def test_features(datasets_path, figures_path):
    """Generate correlation heatmaps for continuous variables in multiple datasets."""
    
    datasets = ['_ds1', '_ds2']
    
    # Define all prefixes for continuous variables
    prefixes = (
        'minutes_played_', 'fg_pct_', 'three_pct_', 'ft_pct_',
        'eff_per_point', 'eff_per_min', 'points_per_min',
        'scoring_impact', 'eff_times_minutes', 'scoring_volume'
    )
    
    for ds in datasets:
        df = pd.read_csv(f'{datasets_path}{ds}.csv', index_col='row_id')
        
        # Select columns matching prefixes
        continuous_variables = [col for col in df.columns if col.startswith(prefixes)]
        
        # Heatmap for continuous variables only
        if continuous_variables:
            figure_name = f'continuous_variables{ds}'
            correlation_and_heatmap(df[continuous_variables], figures_path, figure_name)
        
        # Heatmap for all variables
        figure_name = f'data_set{ds}'
        correlation_and_heatmap(df, figures_path, figure_name)
