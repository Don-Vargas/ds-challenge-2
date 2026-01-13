import pandas as pd

def load_data_csv(path):
    df = pd.read_csv(path)
    columns=['target', 'player_id']
    X = df.drop(columns=[columns])
    y = df['target']
    return X, y

def data_loading(path):
    load_data_csv(path)