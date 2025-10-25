# Tests

Tests automatizados y scripts de validación:

## Tests de Integración
- `test_sklearn_pipeline.py`: Test del pipeline sklearn
- `test_full_integration.py`: Validación completa del sistema (7 tests)

## Validaciones de Estructura
- `validate_cookiecutter.py`: Validación estructura Cookiecutter
- `validate_dataset.py`: Validación módulo dataset
- `validate_features.py`: Validación módulo features
- `validate_plots.py`: Validación módulo plots

## Ejecutar Tests
```bash
# Test individual
python tests/test_sklearn_pipeline.py

# Test completo
python tests/test_full_integration.py

# Validaciones
python tests/validate_cookiecutter.py
```
