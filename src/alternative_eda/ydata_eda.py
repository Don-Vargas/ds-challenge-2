import os
import mlflow
import pandas as pd
from ydata_profiling import ProfileReport
from src.utils.storage import load_pickle

def save_profile_report(df: pd.DataFrame, output_path: str, report_name: str):
    """
    Generate a ydata-profiling report and save it as HTML.
    """
    os.makedirs(output_path, exist_ok=True)
    report = ProfileReport(df, title=report_name, explorative=True)
    report_file = os.path.join(output_path, f"{report_name}.html")
    report.to_file(report_file)
    print(f"Saved profile report: {report_file}")
    return report_file

def run_eda():
    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------
    version = 'v2'
    all_rankings_file = f'training_parameter_results/{version}/all_rankings.pkl'
    feta_dict = load_pickle('src/eda/eda_feature_engineered.pkl')
    
    df = feta_dict["features"].copy()
    df["player_id"] = feta_dict["player_id"]
    df["target"] = feta_dict["target"]

    output_dir = f'src/eda/reports/{version}/'
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Start MLflow run
    # ------------------------------------------------------------------
    mlflow.set_experiment("EDA_Binary_Classification")
    with mlflow.start_run(run_name=f"EDA_{version}"):
        # Combined analysis
        combined_report = save_profile_report(df, output_dir, "eda_combined")
        mlflow.log_artifact(combined_report, name="eda_reports")

        # Target=0 analysis
        df_0 = df[df["target"] == 0].copy()
        report_0 = save_profile_report(df_0, output_dir, "eda_target_0")
        mlflow.log_artifact(report_0, name="eda_reports")

        # Target=1 analysis
        df_1 = df[df["target"] == 1].copy()
        report_1 = save_profile_report(df_1, output_dir, "eda_target_1")
        mlflow.log_artifact(report_1, name="eda_reports")

        # Log dataset info
        mlflow.log_param("dataset_rows", df.shape[0])
        mlflow.log_param("dataset_columns", df.shape[1])
        mlflow.log_param("target_distribution", df["target"].value_counts().to_dict())

    print(f"EDA completed. Reports saved in {output_dir} and logged to MLflow.")

    return output_dir
