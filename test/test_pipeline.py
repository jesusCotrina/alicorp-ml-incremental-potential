# test_pipeline.py
import pytest
import pandas as pd
import numpy as np

# Datos simulados para testing rápidos de estructura
@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'customer_id': [1, 2, 3],
        'target': [0, 1, 0],
        'CAT1_brecha_ecosistema': [0.5, 1.2, np.nan], # Forzamos un NaN para el test
        'ratio_recencia_monto_3w': [0.1, 0.4, 0.2]
    })

def test_check_missing_values(sample_data):
    """Test 1: Verificar que no existan valores nulos críticos antes de entrenar"""
    # En producción real usarías df_model_v3
    # Aquí validamos si quedan NaNs residuales después de la ingeniería de variables
    nan_count = sample_data['CAT1_brecha_ecosistema'].isna().sum()
    
    # El test fallará a propósito si encuentra nulos
    assert nan_count == 0, f"Error: Se encontraron {nan_count} valores nulos en columnas críticas."

def test_target_distribution(sample_data):
    """Test 2: Validar que el target sea binario (0 y 1)"""
    unique_targets = sample_data['target'].unique()
    assert set(unique_targets).issubset({0, 1}), "Error: El target contiene valores diferentes a 0 y 1."