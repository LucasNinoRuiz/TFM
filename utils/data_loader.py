import pandas as pd
import json

def load_data(data_file):
    """
    Función para cargar los datos del archivo JSON y convertirlos en un DataFrame de pandas.

    Parámetros:
    data_file (str): Ruta al archivo JSON que contiene los datos.

    Retorna:
    pandas.DataFrame: DataFrame de pandas que contiene los datos cargados.
    """
    # Leer el archivo JSON
    with open(data_file, "r") as f:
        materials_data = json.load(f)

    # Convertir la lista de diccionarios en un dataframe de pandas
    df = pd.DataFrame(materials_data.values())

    # Agregar los identificadores de los materiales como nueva columna en el datafram
    df.insert(0, 'material_id', materials_data.keys())

    # Exploración y verificación del formato de los datos
    # Mostrar las primeras 5 filas
    print("Primeras 5 filas del DataFrame:")
    print(df.head())
    
    # Mostrar las últimas 5 filas
    print("\nÚltimas 5 filas del DataFrame:")
    print(df.tail())
    
    # Verificar el tipo de datos de cada columna
    print("\nTipo de datos de cada columna:")
    print(df.dtypes)
    
    # Información general del DataFrame
    print("\nInformación general del DataFrame:")
    print(df.info())
    
    # Descripción estadística
    print("\nDescripción estadística del DataFrame:")
    print(df.describe())
    
    # Verificar si hay valores nulos
    print("\nCantidad de valores nulos en cada columna:")
    print(df.isnull().sum())

    # Mostrar las columnas del DataFrame
    print("\nColumnas del DataFrame:")
    print(df.columns)

    return df