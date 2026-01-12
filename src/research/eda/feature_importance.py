import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.storage import path_validate

sns.set_theme(style="whitegrid")  # nicer plots

# --------------------------
# Functions for feature importance
# --------------------------

def train_random_forest(X, y, n_estimators=500, random_state=42):
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    rf.fit(X, y)
    return rf

def get_tree_importance(model, X):
    importance = pd.Series(model.feature_importances_, index=X.columns)
    return importance.sort_values(ascending=False)

def get_permutation_importance(model, X, y, n_repeats=10, random_state=42):
    result = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=random_state)
    importance = pd.Series(result.importances_mean, index=X.columns)
    return importance.sort_values(ascending=False)

def get_shap_importance(model, X):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)[1]  # For binary classification, class 1
    importance = pd.Series(np.abs(shap_values).mean(axis=0), index=X.columns)
    return importance.sort_values(ascending=False)

def rank_features(X, y):
    """
    Returns a dictionary with Tree, Permutation, SHAP importance.
    Also adds aggregated ranking by averaging ranks.
    """
    model = train_random_forest(X, y)
    
    tree_imp = get_tree_importance(model, X)
    perm_imp = get_permutation_importance(model, X, y)
    shap_imp = get_shap_importance(model, X)
    
    # Aggregate ranking by average rank
    df = pd.DataFrame({
        'tree': tree_imp,
        'permutation': perm_imp,
        'shap': shap_imp
    })
    df['avg_rank'] = df.rank(ascending=False).mean(axis=1)
    df = df.sort_values('avg_rank')
    
    return {
        "tree_importance": tree_imp,
        "permutation_importance": perm_imp,
        "shap_importance": shap_imp,
        "aggregated_ranking": df
    }

# --------------------------
# Plotting function
# --------------------------

def plot_feature_importance(feature_rankings, top_n=10, dataset_name="Dataset", output_path='output_path'):
    """
    Plots the top_n features for Tree, Permutation, SHAP, and Aggregated importance.
    """
    importance_types = ['tree_importance', 'permutation_importance', 'shap_importance']
    
    for imp_type in importance_types:
        imp = feature_rankings[imp_type].head(top_n)
        plt.figure(figsize=(8, 5))
        sns.barplot(x=imp.values, y=imp.index, palette="viridis")
        plt.title(f"{dataset_name} - Top {top_n} Features by {imp_type.replace('_', ' ').title()}")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()

        file_path = f"{output_path}feature_importance_importance/{imp_type}.png"
        path_validate(file_path)
        plt.tight_layout()
        plt.savefig(file_path, dpi=300)
        plt.close()
    
    # Aggregated ranking
    agg = feature_rankings['aggregated_ranking'].head(top_n)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=agg['avg_rank'], y=agg.index, palette="magma")
    plt.title(f"{dataset_name} - Top {top_n} Features by Aggregated Ranking")
    plt.xlabel("Average Rank")
    plt.ylabel("Feature")
    plt.tight_layout()
    
    file_path = f"{output_path}feature_importance_importance/aggregated_ranking.png"
    path_validate(file_path)

    plt.tight_layout()
    plt.savefig(file_path, dpi=300)
    plt.close()

# --------------------------
# Main function
# --------------------------

def ranking_kings(datasets_path, output_path):
    datasets = ['_ds1', '_ds2']
    
    for ds in datasets:
        df = pd.read_csv(f'{datasets_path}{ds}.csv', index_col='row_id')
        
        # Separate features and target
        X = df.drop(columns=['target', 'player_id'])
        y = df['target']
        
        # Rank features
        feature_rankings = rank_features(X, y)
        
        # Plot results
        plot_feature_importance(feature_rankings, top_n=10, dataset_name=ds, output_path=output_path)