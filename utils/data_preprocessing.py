import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def clean_data(data):
    # Eliminar filas con datos faltantes
    cleaned_data = data.dropna()

    # Eliminar filas con datos incoherentes (por ejemplo, valores negativos en características que sólo pueden ser positivas)
    cleaned_data = cleaned_data[cleaned_data["formation_energy_per_atom"] >= 0]
    cleaned_data = cleaned_data[cleaned_data["band_gap"] >= 0]
    cleaned_data = cleaned_data[cleaned_data["density"] >= 0]
    cleaned_data = cleaned_data[cleaned_data["unit_cell_volume"] >= 0]
    cleaned_data = cleaned_data[cleaned_data["mass"] >= 0]
    cleaned_data = cleaned_data[cleaned_data["energy"] >= 0]

    # Eliminar filas duplicadas
    cleaned_data = cleaned_data.drop_duplicates()

    # Convertir datos categóricos en valores numéricos
    cleaned_data["crystal_system"] = cleaned_data["crystal_system"].astype("category").cat.codes
    cleaned_data["structure"] = cleaned_data["structure"].astype("category").cat.codes

    return cleaned_data

def normalize_data(clean_data):
    # Separar características y etiquetas (si es necesario)
    # En este caso, asumimos que no hay una columna de etiquetas en el conjunto de datos
    features = clean_data.copy()

    # Aplicar normalización min-max a las características
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(features)

    # Crear un nuevo DataFrame con los datos normalizados
    normalized_data = pd.DataFrame(normalized_features, columns=features.columns)

    return normalized_data

def split_data(normalized_data, test_size=0.2, random_state=42):
    # Separar los datos limpios y normalizados en un conjunto de entrenamiento y otro de validación
    train_data, val_data = train_test_split(normalized_data, test_size=test_size, random_state=random_state)
    return train_data, val_data

