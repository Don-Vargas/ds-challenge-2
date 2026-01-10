import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# List of categorical features
categorical_features = [
    'game_location_Away', 'steals_0', 'steals_1', 'steals_2', 'steals_3', 
    'blocks_0', 'blocks_1', 'blocks_2', 'blocks_3', 
    'turnovers_0', 'turnovers_1', 'turnovers_2', 'turnovers_3', 
    'turnovers_4', 'turnovers_5'
]
target = ['target']

# Function to perform Chi-Square Test between pairs of categorical variables
def chi_square_test(df, cat_features):
    print("Running Chi-Square Test of Independence between pairs of categorical variables...\n")
    significant_pairs = []
    
    for i in range(len(cat_features)):
        for j in range(i + 1, len(cat_features)):
            feature_1 = cat_features[i]
            feature_2 = cat_features[j]
            # Create a contingency table
            contingency_table = pd.crosstab(df[feature_1], df[feature_2])
            # Perform Chi-Square Test
            chi2, p, _, _ = chi2_contingency(contingency_table)
            if p < 0.05:
                print(f"  --> Significant relationship between {feature_1} and {feature_2} (p < 0.05)\n")

                significant_pairs.append((feature_1, feature_2))  # Add significant pair to list
    # Visualize relationships using stacked bar plots for significant pairs
    stacked_bar_plot(df, categorical_features, significant_pairs)
            #else:
            #    print(f"  --> No significant relationship between {feature_1} and {feature_2} (p >= 0.05)\n")
    
    return significant_pairs

# Function to visualize relationships between categorical variables using stacked bar plots
def stacked_bar_plot(df, cat_features, significant_pairs):
    print("Creating stacked bar plots for categorical features...\n")
    
    for feature_1, feature_2 in significant_pairs:
        # Plotting the stacked bar plot for pairs with significant relationship
        plt.figure(figsize=(10, 6))
        sns.countplot(x=feature_1, data=df, hue=feature_2, dodge=False)
        plt.title(f"Stacked Bar Plot of {feature_1} by {feature_2}")
        plt.xlabel(feature_1)
        plt.ylabel("Count")
        plt.legend(title=feature_2, loc='upper right')
        plt.show()

# Function to visualize the correlation matrix for numeric features
def correlation_test(df, numeric_features):
    print("Creating correlation matrix heatmap for numeric features...\n")
    # Compute the correlation matrix
    correlation_matrix = df[numeric_features].corr()

    # Plotting the heatmap of the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
    plt.title("Correlation Matrix for Numeric Features")
    plt.show()

# Main function to test and visualize categorical and numeric data
def test_features(datasets_path):
    # Read the dataset
    df = pd.read_csv(datasets_path + '_1.csv', index_col='row_id')
    
    # Extract numerical features (excluding categorical and target columns)
    numeric_features = df.drop(columns=categorical_features + target)
    
    # Run numeric correlation test (correlation matrix)
    correlation_test(df, numeric_features.columns)
    
    # Run Chi-Square Test on pairs of categorical features and get significant pairs
    significant_pairs = chi_square_test(df, categorical_features)
    
