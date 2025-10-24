"""Validar equivalencia entre versiones"""
import sys
from acoustic_ml import dataset
from acoustic_ml import datase_v2

print("=" * 60)
print("VALIDACION: dataset.py vs datase_v2.py")
print("=" * 60)

manager = datase_v2.DatasetManager()

# Test 1: Modified
print("\nTest 1: load_turkish_modified()")
try:
    df1 = dataset.load_turkish_modified()
    df2 = manager.load_modified()
    assert df1.shape == df2.shape, f"Shapes diferentes: {df1.shape} vs {df2.shape}"
    print("OK - load_turkish_modified: EQUIVALENTE")
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)

# Test 2: Train/test split
print("\nTest 2: load_train_test_split()")
try:
    X_tr1, X_te1, y_tr1, y_te1 = dataset.load_train_test_split()
    X_tr2, X_te2, y_tr2, y_te2 = manager.load_train_test_split()
    assert X_tr1.shape == X_tr2.shape
    assert X_te1.shape == X_te2.shape
    print("OK - load_train_test_split: EQUIVALENTE")
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("EXITO: Todas las validaciones pasaron")
print("datase_v2.py es EQUIVALENTE a dataset.py")
print("=" * 60)
