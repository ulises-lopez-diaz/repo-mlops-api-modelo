import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import snowflake.connector
import os
from dotenv import load_dotenv
from mlxtend.frequent_patterns import fpgrowth, association_rules
import mlflow
import mlflow.sklearn
import sqlite3
import pickle

# Cargar variables de entorno
load_dotenv()

# Configurar MLflow y crear un nuevo experimento
mlflow.set_experiment("Algoritmos de Recomendación con FP-Growth")

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
    dataframe_retail_data = pd.read_parquet('../data/output_file.parquet', engine='pyarrow')

    # Transformar el DataFrame en formato de cesta de compras
    basket = dataframe_retail_data.groupby(["INVOICENO", "STOCKCODE"])["QUANTITY"].sum().unstack().fillna(0)
    basket = basket.applymap(lambda x: 1 if x > 0 else 0)

    # Usar FP-Growth en lugar de Apriori
    frequent_itemsets = fpgrowth(basket, min_support=0.01, use_colnames=True)

    # Guardar los conjuntos frecuentes en un archivo .pkl
    os.makedirs('models', exist_ok=True)
    frequent_itemsets_path = '../models/frequent_itemsets.pkl'
    with open(frequent_itemsets_path, 'wb') as f:
        pickle.dump(frequent_itemsets, f)

    # Registrar el archivo de conjuntos frecuentes en MLflow
    mlflow.log_artifact(frequent_itemsets_path)

    # Generar las reglas de asociación
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

    # Guardar las reglas de asociación en un archivo .pkl 
    rules_path = '../models/association_rules.pkl'
    with open(rules_path, 'wb') as f:
        pickle.dump(rules, f)

    # Registrar el archivo de reglas de asociación en MLflow
    mlflow.log_artifact(rules_path)

    # Función para recomendar productos basados en reglas de asociación
    def recommend_association_rules(stockcode, rules, n=10):
        product_rules = rules[rules['antecedents'].apply(lambda x: stockcode in x)]
        product_rules = product_rules.sort_values(by='lift', ascending=False).head(n)
        recommended_products = []
        for rule in product_rules['consequents']:
            recommended_products.extend(list(rule))
        return recommended_products

    # Ejemplo de uso
    stockcode = "23355"
    recommendations = recommend_association_rules(stockcode=stockcode, rules=rules)
    
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


