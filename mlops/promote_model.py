import sys
import mlflow
from mlflow.tracking import MlflowClient

def promote_model(target_stage):
    # 1. Configuración de la conexión al servidor centralizado de MLflow
    # Asegúrate de que esta sea la misma URL que usas en train.py
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    client = MlflowClient()
    
    # 2. Nombre exacto del modelo registrado (debe coincidir con tu train.py)
    model_name = "Modelo_Potencial_Alicorp"
    
    try:
        # 3. Buscar todas las versiones registradas del modelo
        versions = client.search_model_versions(f"name='{model_name}'")
        
        if not versions:
            print(f"Error: No se encontró el modelo '{model_name}' en el registro.")
            sys.exit(1)
            
        # 4. Obtener la versión más reciente (la de número más alto)
        latest_version = sorted(versions, key=lambda x: int(x.version))[-1].version
        
        # 5. Realizar la transición de etapa (Staging o Production)
        print(f"Promocionando {model_name} (versión {latest_version}) a la etapa: '{target_stage}'...")
        
        client.transition_model_version_stage(
            name=model_name,
            version=latest_version,
            stage=target_stage,
            archive_existing_versions=True # Esto archiva automáticamente versiones anteriores en el mismo stage
        )
        
        print(f"¡Éxito! El modelo {model_name} v{latest_version} ahora está en '{target_stage}'.")
        
    except Exception as e:
        print(f"Error durante la promoción del modelo: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Validar que se haya pasado un argumento (staging o production)
    if len(sys.argv) < 2:
        print("Uso correcto: python promote_model.py <staging|production>")
        sys.exit(1)
        
    stage = sys.argv[1].lower()
    
    if stage not in ["staging", "production"]:
        print("Error: El argumento debe ser 'staging' o 'production'")
        sys.exit(1)
        
    promote_model(stage)