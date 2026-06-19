# Modelo Predictivo: Potencial Incremental Alicorp

## Descripcion del Proyecto
Este proyecto desarrolla un modelo predictivo para identificar clientes B2B con potencial de crecimiento incremental. La solucion abarca desde el preprocesamiento de datos transaccionales, ingenieria de variables temporales, entrenamiento optimizado con Optuna y un pipeline de MLOps para el despliegue y monitoreo.

## Estructura del Repositorio
- /mlops: Scripts principales de entrenamiento, promocion y despliegue.
- /dataset: Almacenamiento de datos (CSV) y datasets procesados.
- /config: Archivos de configuracion y mejores parametros del modelo.
- /test: Pruebas unitarias para validar el pipeline.
- .github/workflows: Configuracion del CI/CD de GitHub Actions.

## Requisitos Previstos
Asegurate de tener Python 3.10+ instalado. Instala las dependencias:
pip install -r requirements.txt

## Ejecucion de MLflow
El registro de modelos se gestiona mediante MLflow con un backend local en SQLite. Para iniciar el servidor de seguimiento y visualizar tus experimentos:

1. Navega a la carpeta del proyecto.
2. Ejecuta el siguiente comando:
   mlflow ui --backend-store-uri sqlite:///mlflow.db

Esto levantara una interfaz en http://127.0.0.1:5000 donde podras gestionar los modelos registrados.

## Configuracion de GitHub Actions (Self-Hosted Runner)
Para ejecutar el pipeline de CI/CD en tu servidor local o maquina de desarrollo:

1. Ve a tu repositorio en GitHub > Settings > Actions > Runners.
2. Haz clic en "New self-hosted runner".
3. Sigue las instrucciones de descarga segun tu sistema operativo.
4. Antes de iniciar, configura el runner ejecutando el script de configuracion:
   .\config.cmd  # En Windows
5. Una vez configurado, inicia el servicio:
   .\run.cmd

El runner debe mantenerse abierto para que GitHub pueda enviar los trabajos al pipeline.

## Pipeline de Entrenamiento y Despliegue
El proyecto utiliza GitHub Actions para automatizar el ciclo de vida del modelo:

1. Pruebas: Ejecucion automatica de `pytest` sobre el dataset procesado.
2. Entrenamiento: Ejecucion de `train.py` para entrenar y registrar en MLflow.
3. Staging: Promocion automatica a ambiente de pruebas.
4. Produccion: Despliegue protegido por aprobacion manual en el entorno de `production` definido en GitHub.

## Monitoreo
Se utiliza Evidently AI para detectar Data Drift y Concept Drift. Tras cada inferencia batch, se genera un reporte HTML en la carpeta /reports que compara la distribucion actual contra el set de entrenamiento.