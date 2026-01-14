import os
import pickle
import pandas as pd

def path_validate(filepath):
    """
    Si `path` es una carpeta, la crea si no existe.
    Si `path` es un archivo, crea la carpeta contenedora.
    """
    directory = filepath if os.path.splitext(filepath)[1] == "" else os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)

def load_pickle(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)
    print(f"Objeto cargado en {filepath}")

def save_pickle(obj, filepath):
    """
    Guarda un objeto en formato pickle en la ruta indicada.
    Si la carpeta no existe, la crea automáticamente.

    Args:
        obj: Objeto Python a guardar.
        filepath (str): Ruta completa del archivo destino (ej. 'data/scaler/scaler.pkl').
    """
    # Crear carpeta si no existe
    path_validate(filepath)

    # Guardar objeto
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)

    print(f"Objeto guardado en {filepath}")

def export_data(df, output_path):
    df.to_csv(output_path, index=True)

def ingest_data(df_path, index_col, target_col=None):
    df = pd.read_csv(df_path, index_col=index_col)

    if target_col is not None:
        X = df.drop(columns=[target_col])
        y = df[target_col]
    else:
        X = df
        y = None

    return X, y
