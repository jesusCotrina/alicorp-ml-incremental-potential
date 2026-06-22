import sys
import mlflow
from mlflow.tracking import MlflowClient

def promote_model(alias_name):
    # 1. Configuración de la conexión
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    client = MlflowClient()
    
    # 2. Nombre del modelo
    model_name = "Modelo_Potencial_Alicorp"
    
    try:
        # 3. Buscar todas las versiones y obtener la última
        versions = client.search_model_versions(f"name='{model_name}'")
        
        if not versions:
            print(f"Error: No se encontró el modelo '{model_name}' en el registro.")
            sys.exit(1)
            
        latest_version = sorted(versions, key=lambda x: int(x.version))[-1].version
        
        # 4. Asignar el Alias (La nueva forma recomendada)
        print(f"Asignando alias '{alias_name}' a {model_name} (versión {latest_version})...")
        
        client.set_registered_model_alias(
            name=model_name,
            alias=alias_name,
            version=latest_version
        )
        
        print(f"¡Éxito! Alias '{alias_name}' apuntando a {model_name} v{latest_version}.")
        
    except Exception as e:
        print(f"Error durante la asignación del alias: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso correcto: python promote_model.py <staging|production>")
        sys.exit(1)
        
    alias = sys.argv[1].lower()
    
    # Nota: Los aliases pueden ser cualquier string, pero mantenemos tu convención
    if alias not in ["staging", "production"]:
        print("Error: El alias debe ser 'staging' o 'production'")
        sys.exit(1)
        
    promote_model(alias)