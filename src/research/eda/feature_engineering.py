import pandas as pd
from src.artifacts.feature_engineering_relationships import (age_minutes_played, 
                                                             efficiency_minutes_played_relationship, 
                                                             efficiency_points_relationship,
                                                             points_minutes_played_relationship)

def feature_engineering_pipeline(df_path, data_set_path):
    df = pd.read_csv(df_path, index_col='row_id')

    df = age_minutes_played.engineering_pipeline(df)
    df = efficiency_points_relationship.engineering_pipeline(df)
    df = efficiency_minutes_played_relationship.engineering_pipeline(df)
    df = points_minutes_played_relationship.engineering_pipeline(df)

    output_path = f"{data_set_path}_feature_engineering.csv"
    df.to_csv(output_path, index=True)

    return output_path
