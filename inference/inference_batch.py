import mlflow
import pandas as pd

def run_batch_inference():
    # 1. Cargar el modelo desde el registro
    model_uri = "models:/Modelo_Potencial_Alicorp/latest"
    model = mlflow.xgboost.load_model(model_uri)
    
    # 2. Leer nuevos datos (Batch diario/semanal)
    df_nuevos = pd.read_csv("../dataset/nuevos_clientes_semana.csv")
    
    # 3. Preprocesar igual que en el entrenamiento
    # (¡IMPORTANTE! Debes mantener la misma lógica de dummies)
    X_new = df_nuevos.drop(columns=["customer_id"])
    X_new = pd.get_dummies(X_new, columns=["segment", "territory_id"])
    
    # 4. Predecir
    predictions = model.predict_proba(X_new)[:, 1]
    
    # 5. Guardar resultados para negocio
    df_nuevos["score_potencial"] = predictions
    df_nuevos.to_csv("../outputs/predicciones_batch.csv", index=False)
    print("Inferencia batch completada con éxito.")