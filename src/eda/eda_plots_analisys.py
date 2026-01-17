from src.utils.storage import (
    ingest_data,
    export_data,
    save_pickle,
    load_pickle
)
from src.research import (
    eda,
    plot_distributions,
    correlation,
    plot_importance
)


def analyze():
    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------
    version = 'v1'
    all_rankings_file = f'training_parameter_results/{version}/all_rankings.pkl'
    all_rankings = load_pickle(all_rankings_file)
    ds = load_pickle('src/eda/eda_ds_dictionary.pkl')
    feta_dict = load_pickle('src/eda/eda_feature_engineered.pkl')

    # ------------------------------------------------------------------
    # feature engineered
    # ------------------------------------------------------------------
    df = feta_dict["features"].copy()
    df["player_id"] = feta_dict["player_id"]
    df["target"] = feta_dict["target"]
    plot_distributions.generate_plots(df,'src/eda/figures/distributions/')

    eda.pandas_summary(df, 'src/eda/tables/eda_engineered_ds.csv')
    correlation.continuous_features_correlation_analysis(ds,'src/eda/figures/')

    # ------------------------------------------------------------------
    # all rankins
    # ------------------------------------------------------------------
    print('--__--__'*50)
    for dataset_name, dataset in all_rankings.items():
        plot_importance.plot_feature_importance(
            dataset, 
            10,
            dataset_name,
            'src/eda/figures/feature_importance/',
            )
