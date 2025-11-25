
#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#
import os
import gzip
import json
import pickle
import pandas as pd
from glob import glob
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error

# --- CONFIGURACIÓN GLOBAL ---
# Rutas de archivos
INPUT_TRAIN = "files/input/train_data.csv.zip"
INPUT_TEST = "files/input/test_data.csv.zip"
MODEL_DIR = "files/models/"
MODEL_PATH = "files/models/model.pkl.gz"
METRICS_DIR = "files/output/"
METRICS_PATH = "files/output/metrics.json"

# Configuración del Preprocesamiento (Lógica Original)
TARGET_COL = "Present_Price"  # Variable a pronosticar según lógica original
DROP_COLS = ['Year', 'Car_Name']
CATEGORICAL_COLS = ['Fuel_Type', 'Selling_type', 'Transmission'] # Nombres exactos del CSV


def load_data(filepath):
    """
    Carga el dataset desde un archivo comprimido zip.
    """
    dataframe = pd.read_csv(
        filepath,
        index_col=False,
        compression="zip",
    )
    return dataframe


def clean_data(df):
    """
    Paso 1: Preprocesamiento de datos.
    - Calcula la edad del vehículo.
    - Elimina columnas irrelevantes.
    """
    df_copy = df.copy()
    current_year = 2021
    
    # Feature Engineering
    df_copy["Age"] = current_year - df_copy["Year"]
    
    # Limpieza
    df_copy = df_copy.drop(columns=DROP_COLS)
    
    return df_copy


def split_data(df):
    """
    Paso 2: Divide el dataset en matriz de características (X) y vector objetivo (y).
    """
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y


def make_pipeline(x_train):
    """
    Paso 3: Crea el pipeline de procesamiento y modelado.
    """
    # Identificar columnas numéricas dinámicamente excluyendo las categóricas definidas
    numerical_features = [col for col in x_train.columns if col not in CATEGORICAL_COLS]

    # Definir transformador de columnas
    preprocessor = ColumnTransformer(
        transformers=[
            # OneHotEncoder para categóricas
            ('cat', OneHotEncoder(), CATEGORICAL_COLS),
            # MinMaxScaler para numéricas
            ('scaler', MinMaxScaler(), numerical_features),
        ],
    )

    # Construir Pipeline completo
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ('feature_selection', SelectKBest(f_regression)),
            ('classifier', LinearRegression())
        ]
    )
    return pipeline


def create_estimator(pipeline):
    """
    Paso 4: Configura la búsqueda de hiperparámetros (GridSearchCV).
    """
    # Malla de parámetros original
    param_grid = {
        'feature_selection__k': range(1, 25), # Rango original
        'classifier__fit_intercept': [True, False],
        'classifier__positive': [True, False]
    }

    # Configuración de GridSearchCV
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=10,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        refit=True,
        verbose=1
    )

    return grid_search


def save_model(path, estimator):
    """
    Paso 5: Guarda el modelo serializado y comprimido con gzip.
    """
    # Gestión de directorios
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Guardado del modelo
    with gzip.open(path, "wb") as f:
        pickle.dump(estimator, f)


def calculate_metrics(dataset_type, y_true, y_pred):
    """
    Paso 6: Calcula las métricas de desempeño solicitadas.
    """
    return {
        "type": "metrics",
        "dataset": dataset_type,
        'r2': float(r2_score(y_true, y_pred)),
        'mse': float(mean_squared_error(y_true, y_pred)),
        'mad': float(median_absolute_error(y_true, y_pred)),
    }


def _run_jobs():
    """
    Función orquestadora principal.
    Ejecuta el flujo completo de carga, limpieza, entrenamiento y evaluación.
    """
    print("Cargando datos...")
    data_train = load_data(INPUT_TRAIN)
    data_test = load_data(INPUT_TEST)

    print("Limpiando datos...")
    data_train = clean_data(data_train)
    data_test = clean_data(data_test)

    print("Dividiendo datos...")
    x_train, y_train = split_data(data_train)
    x_test, y_test = split_data(data_test)

    print("Construyendo pipeline...")
    pipeline = make_pipeline(x_train)

    print("Ejecutando GridSearch y Entrenando...")
    estimator = create_estimator(pipeline)
    estimator.fit(x_train, y_train)

    print(f"Guardando modelo en {MODEL_PATH}...")
    save_model(MODEL_PATH, estimator)

    print("Calculando métricas...")
    y_test_pred = estimator.predict(x_test)
    test_metrics = calculate_metrics("test", y_test, y_test_pred)
    
    y_train_pred = estimator.predict(x_train)
    train_metrics = calculate_metrics("train", y_train, y_train_pred)

    print(f"Guardando métricas en {METRICS_PATH}...")
    os.makedirs(METRICS_DIR, exist_ok=True)
    
    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        file.write(json.dumps(train_metrics) + "\n")
        file.write(json.dumps(test_metrics) + "\n")
        
    print("Proceso finalizado.")


if __name__ == "__main__":
    _run_jobs()