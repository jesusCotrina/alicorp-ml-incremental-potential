from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os

import mlflow
import pandas as pd
import logging

def run_batch_inference():
    logging.info("Iniciando proceso de inferencia batch...")
    
    # 1. Cargar el modelo desde el registro usando alias
    model_uri = "models:/Modelo_Potencial_Alicorp@production"
    
    try:
        model = mlflow.xgboost.load_model(model_uri)
        logging.info("Modelo cargado exitosamente.")
    except Exception as e:
        logging.error(f"Error cargando el modelo: {e}")
        raise e  # Lanzar excepción para que Airflow marque la tarea como fallida
    
    # 2. Leer nuevos datos
    # Ajusta la ruta a una absoluta para evitar problemas con el directorio de trabajo de Airflow
    df_nuevos = pd.read_csv("/ruta/absoluta/al/proyecto/dataset/nuevos_clientes_semana.csv")
    
    # 3. Preprocesamiento
    X_new = df_nuevos.drop(columns=["customer_id"])
    
    # 4. Predecir
    predictions = model.predict_proba(X_new)[:, 1]
    
    # 5. Guardar resultados
    df_nuevos["score_potencial"] = predictions
    df_nuevos.to_csv("/outputs/predicciones_batch_semana.csv", index=False)
    
    logging.info("Inferencia batch completada exitosamente.")

# Agregamos la ruta de tus scripts al path para poder importarlos
sys.path.append(os.path.abspath("/airflow/dags/mlops"))

default_args = {
    'owner': 'mlops_team',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'alicorp_batch_inference_weekly',
    default_args=default_args,
    description='Inferencia semanal del modelo de potencial incremental',
    schedule_interval='@weekly', # Se ejecuta una vez por semana
    catchup=False
) as dag:

    tarea_inferencia = PythonOperator(
        task_id='ejecutar_batch_inference',
        python_callable=run_batch_inference
    )

tarea_inferencia






# Ya no necesitas el if __name__ == "__main__" si Airflow importa la función