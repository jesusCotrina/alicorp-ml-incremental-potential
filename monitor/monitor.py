
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset

import pandas as pd
import joblib

# 1. Cargar el dataset de entrenamiento (el que usaste para entrenar)
# Debes tenerlo guardado desde tu archivo train.py
X_train = pd.read_csv("../dataset/data_cliente.csv") 

# 2. Leer tus datos nuevos (estos ya pasaron por tu pipeline de inferencia)
# Asegúrate de que X_nuevos tenga EXACTAMENTE las mismas columnas que X_train
X_nuevos = pd.read_csv("../dataset/nuevos_clientes.csv")

# 3. Validación rápida de columnas (Muy importante para el monitoreo)
if not list(X_train.columns) == list(X_nuevos.columns):
    print("¡Alerta! Las columnas no coinciden. El monitoreo fallará.")
else:
    print("Datos listos para monitorear.")

# 1. Preparas tu reporte de comparación
data_drift_report = Report(metrics=[
    DataDriftPreset(), # Detecta si tus features cambiaron de distribución
    TargetDriftPreset() # Detecta si las predicciones del modelo cambiaron su comportamiento
])

# 2. Corres el análisis
data_drift_report.run(
    reference_data=X_train, # Datos de cuando el modelo era "perfecto"
    current_data=X_nuevos    # Datos que acabas de inferir hoy
)

# 3. Lo guardas como un archivo HTML local
data_drift_report.save_html("reporte_monitoreo_batch.html")
