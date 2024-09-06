import os
from dotenv import load_dotenv
import sqlite3
import mlflow
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
import pickle
import snowflake.connector

def check_file_exists(file_path):
    """
    Verifica si un archivo existe en la ruta dada.

    Args:
        file_path (str): Ruta al archivo que se desea verificar.

    Returns:
        bool: True si el archivo existe, False en caso contrario.
    """
    return os.path.exists(file_path)

def connect_to_snowflake():
    """
    Conecta a la base de datos Snowflake y devuelve la conexión.

    Returns:
        snowflake.connector.SnowflakeConnection: Conexión a la base de datos Snowflake.
    """
    conn = snowflake.connector.connect(
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        warehouse=os.getenv("DESTINATION_SNOWFLAKECREDENTIALS_WAREHOUSE"),
        database=os.getenv("DESTINATION_SNOWFLAKECREDENTIALS_DATABASE_GOLD"),
        schema=os.getenv("DESTINATION_SNOWFLAKECREDENTIALS_SCHEMA")
    )
    return conn

def fetch_data_from_snowflake(conn, query=""):
    """
    Ejecuta una consulta en Snowflake y devuelve los datos como un DataFrame.

    Args:
        conn (snowflake.connector.SnowflakeConnection): Conexión a Snowflake.
        query (str): Consulta SQL a ejecutar.

    Returns:
        pandas.DataFrame: DataFrame con los datos obtenidos de Snowflake.
    
    Raises:
        snowflake.connector.errors.ProgrammingError: Si hay un error en la consulta SQL.
        Exception: Si ocurre un error al ejecutar la consulta.
    """
    if query is None:
        return None

    try:
        cursor = conn.cursor()
        cursor.execute(query)
        dataframe = cursor.fetch_pandas_all()
        cursor.close()
        return dataframe
    except snowflake.connector.errors.ProgrammingError as e:
        print(f"Error de SQL en Snowflake: {e}")
        raise
    except Exception as e:
        print(f"Error al ejecutar la consulta: {e}")
        raise

def load_environment_variables():
    """
    Carga las variables de entorno desde el archivo .env.

    Utiliza la función `load_dotenv` del módulo `dotenv` para cargar las 
    variables de entorno definidas en el archivo .env en el entorno de 
    ejecución actual.
    """
    load_dotenv()

def connect_to_db(db_name='recommendations.db'):
    """
    Conecta a la base de datos SQLite y devuelve la conexión.

    Args:
        db_name (str): Nombre del archivo de la base de datos SQLite. Por defecto es 'recommendations.db'.

    Returns:
        sqlite3.Connection: Conexión a la base de datos SQLite.
    """
    return sqlite3.connect(db_name)

def create_recommendations_table(cursor):
    """
    Crea la tabla de recomendaciones en la base de datos SQLite si no existe.

    Args:
        cursor (sqlite3.Cursor): Cursor de la conexión a la base de datos.
    """
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS recommendations (
            stockcode TEXT,
            recommended_stockcodes TEXT
        )
    ''')

def save_recommendations_to_db(cursor, stockcode, recommendations):
    """
    Guarda las recomendaciones en la base de datos SQLite.

    Args:
        cursor (sqlite3.Cursor): Cursor de la conexión a la base de datos.
        stockcode (str): Código del producto para el cual se generan las recomendaciones.
        recommendations (list of str): Lista de códigos de productos recomendados.
    """
    cursor.execute('''
        INSERT INTO recommendations (stockcode, recommended_stockcodes)
        VALUES (?, ?)
    ''', (stockcode, ','.join(recommendations)))

def close_db_connection(conn):
    """
    Cierra la conexión a la base de datos SQLite.

    Args:
        conn (sqlite3.Connection): Conexión a la base de datos SQLite.
    """
    conn.close()

def setup_mlflow_experiment(experiment_name="Algoritmos de Recomendación con FP-Growth"):
    """
    Configura el experimento en MLflow.

    Args:
        experiment_name (str): Nombre del experimento en MLflow. Por defecto es "Algoritmos de Recomendación con FP-Growth".
    """
    mlflow.set_experiment(experiment_name)

def start_mlflow_run():
    """
    Inicia una nueva ejecución en MLflow.
    """
    mlflow.start_run()

def end_mlflow_run():
    """
    Finaliza la ejecución actual en MLflow.
    """
    mlflow.end_run()

def log_artifact_to_mlflow(file_path):
    """
    Registra un archivo como un artefacto en MLflow.

    Args:
        file_path (str): Ruta al archivo que se va a registrar.
    """
    mlflow.log_artifact(file_path)

def log_param_to_mlflow(key, value):
    """
    Registra un parámetro en MLflow.

    Args:
        key (str): Nombre del parámetro.
        value (any): Valor del parámetro.
    """
    mlflow.log_param(key, value)

def log_metric_to_mlflow(key, value):
    """
    Registra una métrica en MLflow.

    Args:
        key (str): Nombre de la métrica.
        value (float): Valor de la métrica.
    """
    mlflow.log_metric(key, value)

def load_data(file_path='../data/old_invoice_data.parquet'):
    """
    Carga los datos desde un archivo Parquet.

    Args:
        file_path (str): Ruta al archivo Parquet. Por defecto es '../data/old_invoice_data.parquet'.

    Returns:
        pandas.DataFrame: DataFrame con los datos cargados desde el archivo Parquet.
    """
    return pd.read_parquet(file_path, engine='pyarrow')

def transform_to_basket(dataframe):
    """
    Transforma el DataFrame a un formato de cesta para el análisis de asociaciones.

    Args:
        dataframe (pandas.DataFrame): DataFrame con los datos de transacciones.

    Returns:
        pandas.DataFrame: DataFrame transformado en formato de cesta, con las cantidades sumadas.
    """
    basket = dataframe.groupby(["INVOICENO", "STOCKCODE"])["QUANTITY"].sum().unstack().fillna(0)
    basket = basket.astype(bool)
    return basket

def generate_frequent_itemsets(basket, min_support=0.01):
    """
    Genera conjuntos frecuentes utilizando el algoritmo FP-Growth.

    Args:
        basket (pandas.DataFrame): DataFrame en formato de cesta.
        min_support (float): Soporte mínimo para los conjuntos frecuentes. Por defecto es 0.01.

    Returns:
        pandas.DataFrame: DataFrame con los conjuntos frecuentes generados.
    """
    return fpgrowth(basket, min_support=min_support, use_colnames=True)

def save_pickle(obj, file_path):
    """
    Guarda un objeto en un archivo pickle.

    Args:
        obj (object): Objeto que se va a guardar.
        file_path (str): Ruta al archivo donde se guardará el objeto.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)

def generate_association_rules(frequent_itemsets, metric="lift", min_threshold=1):
    """
    Genera reglas de asociación a partir de los conjuntos frecuentes.

    Args:
        frequent_itemsets (pandas.DataFrame): DataFrame con los conjuntos frecuentes.
        metric (str): Métrica utilizada para evaluar las reglas. Por defecto es "lift".
        min_threshold (float): Umbral mínimo para la métrica. Por defecto es 1.

    Returns:
        pandas.DataFrame: DataFrame con las reglas de asociación generadas.
    """
    return association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)

def recommend_association_rules(stockcode, rules, n=10):
    """
    Genera recomendaciones basadas en las reglas de asociación.

    Args:
        stockcode (str): Código del producto para el cual se generan las recomendaciones.
        rules (pandas.DataFrame): DataFrame con las reglas de asociación.
        n (int): Número máximo de recomendaciones a devolver. Por defecto es 10.

    Returns:
        list of str: Lista de códigos de productos recomendados.
    """
    product_rules = rules[rules['antecedents'].apply(lambda x: stockcode in x)]
    product_rules = product_rules.sort_values(by='lift', ascending=False).head(n)
    recommended_products = []
    for rule in product_rules['consequents']:
        recommended_products.extend(list(rule))
    return recommended_products

def main():
    """
    Función principal para ejecutar el flujo de trabajo de recomendación.

    Realiza los siguientes pasos:
    1. Carga las variables de entorno.
    2. Configura el experimento en MLflow y comienza una nueva ejecución.
    3. Conecta a la base de datos Snowflake y obtiene los datos.
    4. Verifica la existencia del archivo Parquet y guarda los datos.
    5. Conecta a la base de datos SQLite y crea la tabla de recomendaciones si no existe.
    6. Transforma los datos, genera conjuntos frecuentes y reglas de asociación.
    7. Guarda los resultados y los registra en MLflow.
    8. Genera recomendaciones y guarda los resultados en la base de datos.
    9. Registra parámetros y métricas en MLflow.
    10. Maneja excepciones y cierra las conexiones.
    """
    load_environment_variables()
    setup_mlflow_experiment()
    start_mlflow_run()

    conn = None
    try:

        ################################ ELIMINAR ANTES DE SUBIR A PIPELINE DE CI/CD ########################################
        # Consulta SQL para obtener datos de Snowflake
        query = """
        SELECT DISTINCT IFT.INVOICENO AS INVOICENO,
            PD.stockcode AS STOCKCODE,
            PD.DESCRIPTION as DESCRIPTION,
            IFT.unitprice AS UNITPRICE,
            IFT.QUANTITY AS QUANTITY,
            IDD.FECHA_INVOICE AS FECHA_INVOICE
        FROM INVOICE_FACT_TABLE IFT 
        INNER JOIN PRODUCT_DIM PD 
        ON PD.product_id = IFT.product_id
        INNER JOIN COUNTRY_DIM CD 
        ON CD.country_id = IFT.country_id
        INNER JOIN INVOICE_DATE_DIM IDD 
        ON IDD.invoice_date_id = IFT.invoice_date_id
        """
        # Conectar a la base de datos de Snowflake
        snowflake_conn = connect_to_snowflake()
        dataframe_retail_data = fetch_data_from_snowflake(snowflake_conn, query)
        snowflake_conn.close()


        ################################ ELIMINAR ANTES DE SUBIR A PIPELINE DE CI/CD ########################################

        # Verificar si el archivo old_invoice_data.parquet ya existe
        data_folder = '../data/'
        old_file_path = os.path.join(data_folder, 'old_invoice_data.parquet')
        new_file_path = os.path.join(data_folder, 'new_invoice_data.parquet')

        if check_file_exists(old_file_path):
            dataframe_retail_data.to_parquet(path=new_file_path, engine='pyarrow', index=False)
        else:
            dataframe_retail_data.to_parquet(path=old_file_path, engine='pyarrow', index=False)

        # Conectar a la base de datos SQLite
        conn = connect_to_db()
        cursor = conn.cursor()

        # Crear tabla si no existe
        create_recommendations_table(cursor)

        # Transformar datos
        basket = transform_to_basket(dataframe_retail_data)

        # Generar conjuntos frecuentes y reglas de asociación
        frequent_itemsets = generate_frequent_itemsets(basket)
        save_pickle(frequent_itemsets, '../models/frequent_itemsets.pkl')
        log_artifact_to_mlflow('../models/frequent_itemsets.pkl')

        rules = generate_association_rules(frequent_itemsets)
        save_pickle(rules, '../models/association_rules.pkl')
        log_artifact_to_mlflow('../models/association_rules.pkl')

        # Generar recomendaciones
        stockcode = "23355"
        recommendations = recommend_association_rules(stockcode=stockcode, rules=rules)

        # Registrar en MLflow
        log_param_to_mlflow("stockcode", stockcode)
        log_metric_to_mlflow("num_recommendations", len(recommendations))

        # Guardar resultados en la base de datos
        save_recommendations_to_db(cursor, stockcode, recommendations)
        conn.commit()

        print(recommendations)

    except Exception as e:
        log_param_to_mlflow("error", str(e))
        print(f"Se produjo un error: {e}")

    finally:
        if conn:
            close_db_connection(conn)
        end_mlflow_run()

if __name__ == "__main__":
    main()
