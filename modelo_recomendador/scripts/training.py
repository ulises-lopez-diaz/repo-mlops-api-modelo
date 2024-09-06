import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import snowflake.connector
import os
from dotenv import load_dotenv
from mlxtend.frequent_patterns import apriori, association_rules
import mlflow
import mlflow.sklearn
import sqlite3
import pickle

# Cargar variables de entorno
load_dotenv()

# Configurar MLflow y crear un nuevo experimento
mlflow.set_experiment("Algoritmos de Recomendación")

# Inicializar MLflow
mlflow.start_run()

try:
    # Conectar a la base de datos SQLite
    conn = sqlite3.connect('recommendations.db')
    cursor = conn.cursor()

    # Crear tabla si no existe
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS recommendations (
            stockcode TEXT,
            recommended_stockcodes TEXT
        )
    ''')

    # Leer el archivo parquet
    dataframe_retail_data = pd.read_parquet('data/output_file.parquet', engine='pyarrow')

    # Crear una tabla de co-ocurrencia
    co_ocurrence_matrix = dataframe_retail_data.pivot_table(index="INVOICENO", columns="STOCKCODE", aggfunc="size", fill_value=0)

    # Guardar la matriz de co-ocurrencia en un archivo .pkl
    os.makedirs('models', exist_ok=True)
    co_ocurrence_matrix_path = 'models/co_ocurrence_matrix.pkl'
    with open(co_ocurrence_matrix_path, 'wb') as f:
        pickle.dump(co_ocurrence_matrix, f)

    # Registrar el archivo de la matriz en MLflow
    mlflow.log_artifact(co_ocurrence_matrix_path)

    # Calcular similitudes entre productos
    product_similarities = cosine_similarity(co_ocurrence_matrix.T)

    # Guardar la matriz de similitudes en un archivo .pkl
    product_similarities_path = 'models/product_similarities.pkl'
    with open(product_similarities_path, 'wb') as f:
        pickle.dump(product_similarities, f)
    
    # Registrar el archivo de similitudes en MLflow
    mlflow.log_artifact(product_similarities_path)

    # Función para recomendar productos similares
    def recommended_products(stockcode, product_similarities, n=10):
        product_idx = co_ocurrence_matrix.columns.get_loc(stockcode)
        sim_scores = list(enumerate(product_similarities[product_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:n+1]
        similar_products = [co_ocurrence_matrix.columns[i[0]] for i in sim_scores]
        return similar_products

    # Ejemplo de uso
    stockcode = "21937"
    recommendations = recommended_products(stockcode=stockcode, product_similarities=product_similarities)
    
    # Registrar resultados en MLflow
    mlflow.log_param("stockcode", stockcode)
    mlflow.log_metric("num_recommendations", len(recommendations))
    
    # Guardar resultados en la base de datos SQLite
    cursor.execute('''
        INSERT INTO recommendations (stockcode, recommended_stockcodes)
        VALUES (?, ?)
    ''', (stockcode, ','.join(recommendations)))
    conn.commit()

    print(recommendations)

except Exception as e:
    # Registrar excepción si ocurre
    mlflow.log_param("error", str(e))
    raise e

finally:
    # Cerrar la conexión a la base de datos SQLite
    conn.close()
    
    # Finalizar la ejecución de MLflow
    mlflow.end_run()
