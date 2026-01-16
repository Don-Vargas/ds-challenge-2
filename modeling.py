from src.utils.storage import ingest_data, export_data, save_pickle, load_pickle
from src.model_experiments import experiments


def model_training_pipeline(
    training_data_path,
    testing_data_path,
    results_path,
    best_model_path,
    version,
    target_col
):
    # ---------------------------------------------------
    # Load data
    # ---------------------------------------------------
    ds = ['ds1', 'ds2', 'ds3', 'ds4', 'ds5', 'ds6', 'ds7', 'ds8', 'ds9', 'ds10']
    results = {}
    best_model = None
    best_roc_auc = -float("inf")  # or 0

    for name in ds:
        X_train, y_train = ingest_data(f'{training_data_path}{name}.csv', index_col='row_id', target_col=target_col)
        X_test,  y_test  = ingest_data(f'{testing_data_path}{name}.csv',  index_col='row_id', target_col=target_col)

        rows_with_nans = X_test[X_test.isna().any(axis=1)]
        # Print the row indices
        print("Rows with NaNs:")
        print(rows_with_nans.index.tolist())
        
        results[name] = experiments.experiment_results(X_train, y_train, X_test, y_test, version)
        results[name]['dataset_name'] = name
        # Update best model
        if results[name]['test_roc_auc'] > best_roc_auc:
            best_roc_auc = results[name]['test_roc_auc']
            best_model = results[name]

    # ---------------------------------------------------
    # Save All Model results
    # ---------------------------------------------------
    export_path = f"{results_path}{version}/model_experiment_results.pkl"
    save_pickle(results, export_path)

    # ---------------------------------------------------
    # Save Best Model to inference blind data
    # ---------------------------------------------------
    export_path = f"{best_model_path}{version}/best_model_{best_model['dataset_name']}.pkl"
    save_pickle(best_model, export_path)


def model_inference_pipeline(inference_data_path, results_path, version, selected_ds):
    """
    Run inference on new data and save predictions to CSV.
    
    Args:
        inference_data_path (str): Path to the data for inference.
        results_path (str): Folder where results should be saved.
        version (str): Model/version identifier.
        selected_ds (str): Dataset identifier.
    """
    # ---------------------------------------------------
    # Load inference data
    # ---------------------------------------------------
    X_infer, _ = ingest_data(f'{inference_data_path}{selected_ds}.csv', index_col='row_id')

    # Keep original index as a column
    X_infer = X_infer.copy()

    # ---------------------------------------------------
    # Generate predictions (probabilities for positive class)
    # ---------------------------------------------------
    model_dict = load_pickle(f'{results_path}{version}/best_model{selected_ds}.pkl')
    print(model_dict)
    model = model_dict['best_estimator'] 
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_infer.drop(columns='row_id'))[:, 1]
    else:
        y_pred_proba = model.predict(X_infer.drop(columns='row_id'))

    # ---------------------------------------------------
    # Prepare dataframe to export
    # ---------------------------------------------------
    df_results = X_infer[['row_id']].copy()  # only keep row_id
    df_results['prediction'] = y_pred_proba

    # ---------------------------------------------------
    # Save predictions
    # ---------------------------------------------------
    export_path = f"{results_path}{version}/final_inferences.csv"
    export_data(df_results, export_path)  # export the dataframe including row_id

    print(f"Inference completed. Predictions saved to: {export_path}")
