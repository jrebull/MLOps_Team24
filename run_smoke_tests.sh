#!/bin/bash
# 🔥 SMOKE TESTS RUNNER
# =====================
# Ejecuta validación completa post-deploy

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOKE_TEST_FILE="$SCRIPT_DIR/smoke_tests.py"

echo "🔥 Iniciando Smoke Tests..."
echo "=================================================="

# Verificar que docker compose está corriendo
echo ""
echo "📋 Verificando docker compose..."
if ! docker ps | grep -q "fastapi_app"; then
    echo "❌ FastAPI container no está corriendo"
    echo ""
    echo "Solución: Ejecuta primero:"
    echo "  docker compose up"
    exit 1
fi

if ! docker ps | grep -q "mlflow"; then
    echo "❌ MLflow container no está corriendo"
    echo ""
    echo "Solución: Ejecuta primero:"
    echo "  docker compose up"
    exit 1
fi

echo "✓ Contenedores detectados"

# Ejecutar smoke tests
echo ""
echo "🚀 Ejecutando tests..."
python3 "$SMOKE_TEST_FILE"

exit_code=$?

echo ""
echo "=================================================="
if [ $exit_code -eq 0 ]; then
    echo "✅ Todos los tests pasaron - Listo para producción"
else
    echo "❌ Algunos tests fallaron - Revisar reporte"
fi

exit $exit_code
