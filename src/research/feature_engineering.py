from src.artifacts.feature_engineering_relationships import (age_minutes_played, 
                                                             efficiency_minutes_played_relationship, 
                                                             efficiency_points_relationship,
                                                             points_minutes_played_relationship)

def feature_creation_pipeline(df):

    df = age_minutes_played.engineering_pipeline(df)
    df = efficiency_points_relationship.engineering_pipeline(df)
    df = efficiency_minutes_played_relationship.engineering_pipeline(df)
    df = points_minutes_played_relationship.engineering_pipeline(df)

    return df
