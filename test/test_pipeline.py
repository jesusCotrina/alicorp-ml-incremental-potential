import pytest
import pandas as pd
import numpy as np

# Definimos las columnas que el modelo SIEMPRE espera
EXPECTED_COLUMNS = ['CAT1_brecha_ecosistema', 'total_neto_cliente', 'target']

@pytest.fixture
def data_procesada():
    # Simulamos el dataset que sale de tu pipeline de transformación
    return pd.read_csv("../dataset/data_procesada.csv")

def test_data_contract(data_procesada):
    """Test 1: Verifica que las columnas obligatorias existan"""
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in data_procesada.columns]
    assert len(missing_cols) == 0, f"Error: Faltan columnas en el dataset: {missing_cols}"

def test_data_ranges(data_procesada):
    """Test 2: Verifica que variables críticas no sean negativas"""
    assert (data_procesada['total_neto_cliente'] >= 0).all(), "Error: Existen montos de venta negativos"

def test_no_nulls(data_procesada):
    """Test 3: Asegura que tras el pipeline, no haya nulos"""
    assert data_procesada[EXPECTED_COLUMNS].isnull().sum().sum() == 0, "Error: Hay valores nulos después del pipeline"
