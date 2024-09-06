import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from dotenv import load_dotenv

import snowflake.connector
import os


def load_environment_variables():
    """
    Carga las variables de entorno desde el archivo .env.

    Utiliza la función `load_dotenv` del módulo `dotenv` para cargar 
    las variables de entorno definidas en el archivo .env en el entorno 
    de ejecución actual.
    """
    load_dotenv()

def check_file_exists(file_path):
    """
    Verifica si un archivo existe en la ruta dada.

    Args:
        file_path (str): La ruta al archivo a verificar.

    Returns:
        bool: True si el archivo existe, False en caso contrario.
    """
    return os.path.exists(file_path)

def connect_to_snowflake():
    """
    Conecta a la base de datos Snowflake y devuelve la conexión.

    Utiliza las variables de entorno para obtener las credenciales necesarias 
    para la conexión a Snowflake.

    Returns:
        snowflake.connector.SnowflakeConnection: La conexión a la base de 
        datos Snowflake.
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

    Parámetros:
    - conn: Conexión a Snowflake.
    - query: Consulta SQL a ejecutar.

    Retorna:
    - Un DataFrame con los datos obtenidos de Snowflake.
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


def detectar_drift():
    """
    Detecta el drift de datos entre dos conjuntos de datos almacenados en archivos Parquet.

    Este script compara un conjunto de datos de referencia con un nuevo conjunto de datos
    para identificar posibles drifts en los datos. Si se detecta un drift, se registra en
    un archivo de texto. De lo contrario, también se registra la ausencia de drift.

    Pasos:
    1. Carga los datos de referencia y los nuevos datos desde archivos Parquet.
    2. Crea un reporte de drift utilizando Evidently.
    3. Ejecuta el reporte excluyendo la columna "Outcome".
    4. Convierte el reporte a un diccionario para extraer el resultado.
    5. Verifica si se detectó drift en los datos.
    6. Guarda el resultado en un archivo de texto.

    Requisitos:
    - pandas
    - evidently

    Retorno:
    - Un archivo de texto que indica si se detectó drift o no.

    """
    load_environment_variables()
    conn = None

    try:


        
         # Conectar a la base de datos de Snowflake
        snowflake_conn = connect_to_snowflake()
        dataframe_retail_data = fetch_data_from_snowflake(snowflake_conn)
        snowflake_conn.close()

        # Verificar si el archivo old_invoice_data.parquet ya existe
        data_folder = '../data/'
        old_file_path = os.path.join(data_folder, 'old_invoice_data.parquet')
        new_file_path = os.path.join(data_folder, 'new_invoice_data.parquet')

        if check_file_exists(old_file_path):
            dataframe_retail_data.to_parquet(path=new_file_path, engine='pyarrow', index=False)
        else:
            dataframe_retail_data.to_parquet(path=old_file_path, engine='pyarrow', index=False)

        # Cargar los datos de referencia y los nuevos datos desde archivos Parquet
        reference_data = pd.read_parquet("../data/old_invoice_data.parquet")
        new_data = pd.read_parquet("../data/new_invoice_data.parquet")
    except FileNotFoundError as e:
        print(f"Error: No se encontró el archivo: {e.filename}")
        with open("drift_detected.txt", "w") as f:
            f.write("error: archivo_no_encontrado")
        return
    except Exception as e:
        print(f"Error al cargar los datos: {e}")
        with open("drift_detected.txt", "w") as f:
            f.write("error: carga_datos")
        return

    try:
        # Crea un reporte de drift de datos
        data_drift_report = Report(metrics=[DataDriftPreset()])

        # Ejecuta el reporte, excluyendo la columna "Outcome" si está presente
        reference_data_dropped = reference_data.drop("Outcome", axis=1, errors='ignore')
        new_data_dropped = new_data.drop("Outcome", axis=1, errors='ignore')

        data_drift_report.run(
            reference_data=reference_data_dropped,
            current_data=new_data_dropped,
            column_mapping=None
        )
    except KeyError as e:
        print(f"Error: La columna especificada no se encuentra en el DataFrame: {e}")
        with open("drift_detected.txt", "w") as f:
            f.write("error: columna_no_encontrada")
        return
    except Exception as e:
        print(f"Error al ejecutar el reporte de drift: {e}")
        with open("drift_detected.txt", "w") as f:
            f.write("error: reporte_drift")
        return

    try:
        # Convierte el reporte a un diccionario
        report_json = data_drift_report.as_dict()

        # Verifica si se detectó drift en los datos
        drift_detected = report_json["metrics"][0]["result"]["dataset_drift"]

        if drift_detected:
            print("Se detectó drift de datos. Reentrenando el modelo.")
            with open("drift_detected.txt", "w") as f:
                f.write("drift_detected")
        else:
            print("No se detectó drift de datos.")
            with open("drift_detected.txt", "w") as f:
                f.write("no_drift")

    except KeyError as e:
        print(f"Error: Clave no encontrada en el reporte: {e}")
        with open("drift_detected.txt", "w") as f:
            f.write("error: clave_no_encontrada")
    except Exception as e:
        print(f"Error al procesar el reporte: {e}")
        with open("drift_detected.txt", "w") as f:
            f.write("error: procesamiento_reporte")

if __name__ == "__main__":
    detectar_drift()
