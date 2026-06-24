# Modelo Predictivo: Potencial Incremental Alicorp

## Descripción del Proyecto

Este proyecto desarrolla un modelo predictivo para identificar clientes B2B con potencial de crecimiento incremental. La solución abarca desde el preprocesamiento de datos transaccionales, ingeniería de variables temporales, entrenamiento optimizado con Optuna y un pipeline de MLOps para el despliegue y monitoreo.

## Estructura del Repositorio

/mlops: Scripts principales de entrenamiento, promoción y despliegue.

/dataset: Almacenamiento de datos (CSV) y datasets procesados.

/config: Archivos de configuración y mejores parámetros del modelo.

/test: Pruebas unitarias para validar el pipeline.

.github/workflows: Configuración del CI/CD de GitHub Actions.

## Requisitos Previstos

Se requiere tener Python 3.12+ instalado. La instalación de las dependencias se realiza mediante el siguiente comando:

```bash
pip install -r requirements.txt
```

## Instalación y Ejecución de MLflow

El registro de modelos se gestiona mediante MLflow con un backend local en SQLite. Para realizar la instalación de la herramienta y el posterior despliegue del servidor de seguimiento con el fin de visualizar los experimentos, se deben seguir estos pasos:

Instalar la librería en el entorno de trabajo:

```bash
pip install mlflow
```

Navegar a la carpeta del proyecto.

Ejecutar el siguiente comando para iniciar el servidor de seguimiento:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Este proceso levantará una interfaz en http://127.0.0.1:5000 donde se pueden gestionar los modelos registrados de forma centralizada.

## Configuración de GitHub Actions (Self-Hosted Runner)

Para ejecutar el pipeline de CI/CD en un servidor local o máquina de desarrollo:

Acceder al repositorio en GitHub y dirigirse a Settings > Actions > Runners.

Hacer clic en "New self-hosted runner".

Seguir las instrucciones de descarga según el sistema operativo utilizado.

Antes de iniciar, configurar el runner ejecutando el script de configuración:

```bash
.\config.cmd  # En Windows
```

Una vez configurado, iniciar el servicio:

```bash
.\run.cmd
```

El runner debe mantenerse en ejecución para que GitHub pueda enviar los trabajos al pipeline correspondiente.

## Pipeline de Entrenamiento y Despliegue

El proyecto utiliza GitHub Actions para automatizar el ciclo de vida del modelo:

* Pruebas: Ejecución automática de pytest sobre el dataset procesado.
* Entrenamiento: Ejecución de train.py para entrenar y registrar el modelo en MLflow.
* Staging: Promoción automática al ambiente de pruebas.
* Producción: Despliegue protegido por aprobación manual en el entorno de production definido en GitHub.

## Monitoreo

Se utiliza Evidently AI para detectar Data Drift y Concept Drift. Tras cada inferencia batch, se genera un reporte HTML en la carpeta /reports que compara la distribución actual contra el set de entrenamiento.
