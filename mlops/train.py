import json
import pandas as pd
import mlflow
import mlflow.xgboost
import mlflow.sklearn
import mlflow.lightgbm
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import sys
import io

# Forzar codificación UTF-8 para evitar errores de caracteres especiales en Windows
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')


# Configuración inicial
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Alicorp_Incremental_Potential")

def get_model_and_flavor(params):
    """Fábrica de modelos: Devuelve el modelo instanciado y su librería de MLflow"""
    model_type = params.pop("model_type")
    
    if model_type == "xgboost":
        return XGBClassifier(**params), mlflow.xgboost
    elif model_type == "random_forest":
        return RandomForestClassifier(**params), mlflow.sklearn
    elif model_type == "lightgbm":
        return LGBMClassifier(**params), mlflow.lightgbm
    else:
        raise ValueError(f"Modelo {model_type} no soportado")

def run_training():
    # 1. Cargar datos
    df_model_v3 = pd.read_csv("../dataset/data_procesada.csv") 
    X = df_model_v3.drop(columns=["customer_id", "target","trend_direction"])
    y = df_model_v3["target"]
    # X = pd.get_dummies(X, columns=["segment", "territory_id"], drop_first=True)
    
    # 2. Split y Desbalanceo
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    ratio_desbalanceo = (y_train == 0).sum() / (y_train == 1).sum()

    # 3. Cargar hiperparámetros ganadores
    with open("../config/best_params.json", "r") as f:
        params = json.load(f)
    
    # Ajustes automáticos según el modelo
    params["scale_pos_weight"] = ratio_desbalanceo
    params["random_state"] = 42
    
    # Obtener modelo dinámico
    model, mlflow_flavor = get_model_and_flavor(params.copy())

    # 4. Entrenamiento con MLflow
    with mlflow.start_run(run_name=f"Entrenamiento_{params.get('model_type', 'modelo')}") as run:
        # Registrar parámetros
        mlflow.log_params(params)
        
        # Entrenar
        model.fit(X_train, y_train)
        
        # Evaluar
        preds = model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, preds)
        
        # Registrar métricas y artefactos
        mlflow.log_metric("auc_test", auc_score)
        mlflow.log_dict({"features": list(X.columns)}, "features.json")
        
        # Guardado dinámico en MLflow
        model_info = mlflow_flavor.log_model(
            model, 
            artifact_path="model",
            registered_model_name="Modelo_Potencial_Alicorp"
        )
        
        # Guardar info de ejecución
        with open("latest_run.txt", "w") as f:
            f.write(f"{model_info.run_id},{model_info.model_uri}")
            
        print(f"Éxito! Modelo registrado. AUC: {auc_score:.4f}")

if __name__ == "__main__":
    run_training()