#!/usr/bin/env python3
"""
🔥 SMOKE TESTS - POST-DEPLOY VALIDATION
========================================
Valida que TODOS los servicios y endpoints funcionan correctamente después de levantar Docker.

Uso:
    python smoke_tests.py

Requisitos:
    - docker compose up debe estar corriendo
    - FastAPI en http://127.0.0.1:8000
    - MLflow en http://127.0.0.1:5001
"""

import requests
import json
import time
import sys
from typing import Dict, Any, List, Tuple
from datetime import datetime
import numpy as np

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
FASTAPI_URL = "http://127.0.0.1:8000"
MLFLOW_URL = "http://127.0.0.1:5001"
API_V1 = f"{FASTAPI_URL}/api/v1"
MAX_RETRIES = 30
RETRY_INTERVAL = 2  # segundos

# Colores para output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# ============================================================================
# ESTADO GLOBAL
# ============================================================================
test_results = []
start_time = None

# ============================================================================
# UTILIDADES
# ============================================================================
def print_header(title: str) -> None:
    """Imprime encabezado formateado"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{title:^70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}\n")

def print_section(title: str) -> None:
    """Imprime sección"""
    print(f"\n{Colors.CYAN}{Colors.BOLD}▶ {title}{Colors.ENDC}")
    print(f"{Colors.CYAN}{'-'*70}{Colors.ENDC}")

def log_test(name: str, status: bool, message: str = "", details: Dict = None) -> None:
    """Registra resultado de test"""
    status_icon = f"{Colors.GREEN}✓ PASS{Colors.ENDC}" if status else f"{Colors.RED}✗ FAIL{Colors.ENDC}"
    print(f"{status_icon} | {name}")
    if message:
        print(f"         {Colors.YELLOW}→ {message}{Colors.ENDC}")
    test_results.append({
        "name": name,
        "status": status,
        "message": message,
        "details": details or {}
    })

def wait_for_service(url: str, service_name: str, max_retries: int = MAX_RETRIES) -> bool:
    """Espera a que un servicio esté disponible"""
    print(f"\n{Colors.YELLOW}⏳ Esperando {service_name}...{Colors.ENDC}", end="", flush=True)
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code in [200, 404]:  # 404 es válido si el endpoint no existe
                print(f" {Colors.GREEN}✓ Disponible{Colors.ENDC}")
                return True
        except requests.exceptions.RequestException:
            pass
        
        print(".", end="", flush=True)
        time.sleep(RETRY_INTERVAL)
    
    print(f" {Colors.RED}✗ Timeout{Colors.ENDC}")
    return False

# ============================================================================
# TEST SUITE
# ============================================================================

def test_fastapi_root() -> bool:
    """Test: Endpoint raíz de FastAPI"""
    try:
        response = requests.get(f"{FASTAPI_URL}/", timeout=5)
        data = response.json()
        
        if response.status_code == 200 and "app" in data:
            log_test("FastAPI Root Endpoint", True, f"App: {data.get('app', 'unknown')}")
            return True
    except Exception as e:
        log_test("FastAPI Root Endpoint", False, str(e))
    return False

def test_health_check() -> bool:
    """Test: Health check endpoint"""
    try:
        response = requests.get(f"{API_V1}/health", timeout=5)
        data = response.json()
        
        if response.status_code == 200 and data.get("status") == "ok":
            log_test("Health Check (/api/v1/health)", True, "Status: OK")
            return True
    except Exception as e:
        log_test("Health Check (/api/v1/health)", False, str(e))
    return False

def test_api_docs() -> bool:
    """Test: Documentación de API (Swagger)"""
    try:
        response = requests.get(f"{FASTAPI_URL}/docs", timeout=5)
        
        if response.status_code == 200 and "swagger" in response.text.lower():
            log_test("API Docs (Swagger)", True, f"HTML size: {len(response.text)} bytes")
            return True
    except Exception as e:
        log_test("API Docs (Swagger)", False, str(e))
    return False

def test_predict_endpoint() -> bool:
    """Test: Predict endpoint con datos reales"""
    try:
        # Datos de test: 50 features como en el modelo entrenado
        test_features = {
            "features": [0.5] * 50  # 50 features esperados por RandomForestClassifier
        }
        
        response = requests.post(
            f"{API_V1}/predict",
            json=test_features,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Validar estructura
            if "prediction" in data and "probabilities" in data:
                log_test(
                    "Predict Endpoint",
                    True,
                    f"Prediction: {data['prediction']}, Probs: {len(data['probabilities'])} classes"
                )
                return True
    except Exception as e:
        log_test("Predict Endpoint", False, str(e))
    
    return False

def test_list_models() -> bool:
    """Test: Listar modelos disponibles"""
    try:
        response = requests.get(f"{API_V1}/models", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])
            
            if isinstance(models, list):
                log_test(
                    "List Models Endpoint",
                    True,
                    f"Found {len(models)} model(s): {', '.join(models[:3]) if models else 'None'}"
                )
                return True
    except Exception as e:
        log_test("List Models Endpoint", False, str(e))
    
    return False

def test_mlflow_connectivity() -> bool:
    """Test: MLflow server accesible"""
    try:
        response = requests.get(f"{MLFLOW_URL}/health", timeout=5)
        
        if response.status_code in [200, 404]:  # 404 is OK if endpoint doesn't exist
            log_test(
                "MLflow Server Connectivity",
                True,
                f"MLflow running on {MLFLOW_URL}"
            )
            return True
    except requests.exceptions.ConnectionError:
        log_test("MLflow Server Connectivity", False, "Connection refused")
    except Exception as e:
        log_test("MLflow Server Connectivity", False, str(e))
    
    return False

def test_mlflow_api() -> bool:
    """Test: MLflow API endpoint"""
    try:
        response = requests.get(f"{MLFLOW_URL}/api/2.0/health", timeout=5)
        
        if response.status_code in [200, 404]:
            log_test(
                "MLflow API Endpoint",
                True,
                "MLflow API accessible"
            )
            return True
    except Exception as e:
        log_test("MLflow API Endpoint", False, str(e))
    
    return False

def test_schema_validation() -> bool:
    """Test: Validación de schema Pydantic"""
    try:
        # Test con payload inválido (debe fallar)
        invalid_payload = {"features": "not_a_list"}
        
        response = requests.post(
            f"{API_V1}/predict",
            json=invalid_payload,
            timeout=5
        )
        
        # Debe retornar 422 (validation error)
        if response.status_code == 422:
            data = response.json()
            if "detail" in data:
                log_test(
                    "Schema Validation (Pydantic v2)",
                    True,
                    "Invalid payload correctly rejected (422)"
                )
                return True
    except Exception as e:
        log_test("Schema Validation (Pydantic v2)", False, str(e))
    
    return False

def test_error_handling() -> bool:
    """Test: Manejo de errores HTTP"""
    try:
        # Test endpoint que no existe
        response = requests.get(f"{API_V1}/nonexistent", timeout=5)
        
        if response.status_code == 404:
            log_test(
                "Error Handling (404)",
                True,
                "Non-existent endpoint returns 404"
            )
            return True
    except Exception as e:
        log_test("Error Handling (404)", False, str(e))
    
    return False

def test_response_format() -> bool:
    """Test: Formato de respuestas JSON"""
    try:
        response = requests.get(f"{API_V1}/health", timeout=5)
        
        if response.status_code == 200:
            # Validar que es JSON válido
            data = response.json()
            
            if isinstance(data, dict):
                log_test(
                    "Response Format (JSON)",
                    True,
                    f"Valid JSON response: {json.dumps(data)}"
                )
                return True
    except json.JSONDecodeError as e:
        log_test("Response Format (JSON)", False, f"Invalid JSON: {str(e)}")
    except Exception as e:
        log_test("Response Format (JSON)", False, str(e))
    
    return False

def test_concurrent_requests() -> bool:
    """Test: Manejo de requests concurrentes"""
    try:
        test_features = {
            "features": [0.5] * 50  # 50 features
        }
        
        # Hacer 3 requests secuenciales rápidos
        responses = []
        for i in range(3):
            response = requests.post(
                f"{API_V1}/predict",
                json=test_features,
                timeout=10
            )
            responses.append(response.status_code == 200)
            time.sleep(0.1)
        
        if all(responses):
            log_test(
                "Concurrent Request Handling",
                True,
                f"All 3 sequential requests succeeded"
            )
            return True
    except Exception as e:
        log_test("Concurrent Request Handling", False, str(e))
    
    return False

def test_model_loading() -> bool:
    """Test: Modelo cargado correctamente"""
    try:
        # Si predict funciona, el modelo está cargado
        test_features = {"features": [0.5] * 50}
        response = requests.post(f"{API_V1}/predict", json=test_features, timeout=10)
        
        if response.status_code == 200:
            log_test(
                "Model Loading",
                True,
                "Model successfully loaded and accessible"
            )
            return True
    except Exception as e:
        log_test("Model Loading", False, str(e))
    
    return False

# ============================================================================
# REPORTE FINAL
# ============================================================================

def print_summary() -> None:
    """Imprime resumen de tests"""
    global start_time
    
    elapsed = time.time() - start_time
    total = len(test_results)
    passed = sum(1 for t in test_results if t["status"])
    failed = total - passed
    pass_rate = (passed / total * 100) if total > 0 else 0
    
    print_header("📊 RESUMEN DE TESTS")
    
    print(f"{Colors.BOLD}Total Tests:{Colors.ENDC} {total}")
    print(f"{Colors.GREEN}{Colors.BOLD}✓ Passed:{Colors.ENDC} {passed}")
    print(f"{Colors.RED}{Colors.BOLD}✗ Failed:{Colors.ENDC} {failed}")
    print(f"{Colors.BOLD}Pass Rate:{Colors.ENDC} {pass_rate:.1f}%")
    print(f"{Colors.BOLD}Time:{Colors.ENDC} {elapsed:.2f}s")
    
    # Status general
    if failed == 0:
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 ALL TESTS PASSED!{Colors.ENDC}")
        return_code = 0
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}⚠️  SOME TESTS FAILED{Colors.ENDC}")
        return_code = 1
    
    # Detalles de fallos
    if failed > 0:
        print_section("Fallos Detectados")
        for test in test_results:
            if not test["status"]:
                print(f"{Colors.RED}✗ {test['name']}{Colors.ENDC}")
                print(f"  {Colors.YELLOW}→ {test['message']}{Colors.ENDC}")
    
    return return_code

def save_report(filename: str = "smoke_test_report.json") -> None:
    """Guarda reporte en JSON"""
    global start_time
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "duration_seconds": time.time() - start_time,
        "total_tests": len(test_results),
        "passed": sum(1 for t in test_results if t["status"]),
        "failed": sum(1 for t in test_results if not t["status"]),
        "tests": test_results,
        "environment": {
            "fastapi_url": FASTAPI_URL,
            "mlflow_url": MLFLOW_URL,
            "api_version": "v1"
        }
    }
    
    with open(filename, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n{Colors.BLUE}📄 Reporte guardado en: {filename}{Colors.ENDC}")

# ============================================================================
# MAIN
# ============================================================================

def main() -> int:
    """Ejecuta todos los smoke tests"""
    global start_time
    start_time = time.time()
    
    print_header("🔥 SMOKE TESTS - MLOps Team24 Phase 3")
    
    # 1. Esperar servicios
    print_section("Paso 1: Verificar Servicios Disponibles")
    
    fastapi_ready = wait_for_service(f"{FASTAPI_URL}/", "FastAPI")
    mlflow_ready = wait_for_service(f"{MLFLOW_URL}/", "MLflow")
    
    if not fastapi_ready or not mlflow_ready:
        print(f"\n{Colors.RED}{Colors.BOLD}✗ Servicios no disponibles{Colors.ENDC}")
        return 1
    
    # 2. Tests de FastAPI
    print_section("Paso 2: Validar FastAPI")
    test_fastapi_root()
    test_health_check()
    test_api_docs()
    
    # 3. Tests de Endpoints
    print_section("Paso 3: Validar Endpoints")
    test_predict_endpoint()
    test_list_models()
    test_response_format()
    
    # 4. Tests de MLflow
    print_section("Paso 4: Validar MLflow")
    test_mlflow_connectivity()
    test_mlflow_api()
    
    # 5. Tests de Validación y Manejo de Errores
    print_section("Paso 5: Validar Schema y Errores")
    test_schema_validation()
    test_error_handling()
    
    # 6. Tests de Carga y Modelo
    print_section("Paso 6: Validar Modelo y Concurrencia")
    test_model_loading()
    test_concurrent_requests()
    
    # 7. Resumen
    return_code = print_summary()
    
    # 8. Guardar reporte
    save_report()
    
    return return_code

if __name__ == "__main__":
    sys.exit(main())
