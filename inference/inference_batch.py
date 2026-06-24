import mlflow
import pandas as pd
import sys
mlflow.set_tracking_uri("http://127.0.0.1:5000")
def run_batch_inference():
    # 1. Cargar el modelo usando el alias 'production'
    # Esto asegura que siempre obtengas el modelo aprobado y no cualquier versión
    model_uri = "models:/Modelo_Potencial_Alicorp@production"
    
    try:
        model = mlflow.xgboost.load_model(model_uri)
    except Exception as e:
        print(f"Error cargando el modelo de producción: {e}")
        sys.exit(1)
    
    # 2. Leer nuevos datos
    df_nuevos = pd.read_csv("../dataset/nuevos_clientes.csv")
    
    # 3. PREPROCESAMIENTO (CRÍTICO)
    # No uses get_dummies aquí si en el entrenamiento tuviste un número diferente de columnas.
    # Lo profesional es usar un Pipeline de Scikit-Learn que guardes en MLflow junto al modelo.
    X_new = df_nuevos.drop(columns=["customer_id", "target","trend_direction"])

    # 4. Predecir
    predictions = model.predict_proba(X_new)[:, 1]
    
    # 5. Guardar resultados
    df_nuevos["score_potencial"] = predictions
    df_nuevos.to_csv("./outputs/predicciones_batch_semana.csv", index=False)
    print("Inferencia batch completada.")

if __name__ == "__main__":
    run_batch_inference()