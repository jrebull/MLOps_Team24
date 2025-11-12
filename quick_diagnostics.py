#!/usr/bin/env python3
"""
⚡ QUICK DIAGNOSTICS - Diagnóstico rápido de servicios
=========================================================

Uso:
    python quick_diagnostics.py

Muestra estado rápido sin esperar retries.
"""

import requests
import sys
from typing import Tuple

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'

def check_endpoint(url: str, name: str, timeout: int = 3) -> Tuple[bool, str]:
    """Verifica si un endpoint está disponible"""
    try:
        response = requests.get(url, timeout=timeout)
        status = response.status_code
        return True, f"✓ {status}"
    except requests.exceptions.Timeout:
        return False, "✗ TIMEOUT"
    except requests.exceptions.ConnectionError:
        return False, "✗ CONNECTION REFUSED"
    except Exception as e:
        return False, f"✗ {str(e)[:30]}"

def main():
    print(f"\n{Colors.CYAN}{Colors.BOLD}⚡ QUICK DIAGNOSTICS{Colors.ENDC}\n")
    
    endpoints = [
        ("http://127.0.0.1:8000/", "FastAPI Root"),
        ("http://127.0.0.1:8000/api/v1/health", "FastAPI Health"),
        ("http://127.0.0.1:8000/docs", "Swagger Docs"),
        ("http://127.0.0.1:5001/", "MLflow Server"),
        ("http://127.0.0.1:5001/api/2.0/health", "MLflow API"),
    ]
    
    all_ok = True
    for url, name in endpoints:
        ok, msg = check_endpoint(url, name)
        color = Colors.GREEN if ok else Colors.RED
        status = "OK" if ok else "FAIL"
        print(f"{color}{Colors.BOLD}{status:4}{Colors.ENDC} | {name:20} | {url}")
        if not ok:
            all_ok = False
    
    print()
    
    # Test predict
    print(f"{Colors.CYAN}Testing Predict Endpoint...{Colors.ENDC}")
    try:
        test_data = {"features": [0.5] * 21}
        response = requests.post(
            "http://127.0.0.1:8000/api/v1/predict",
            json=test_data,
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            if "prediction" in data and "probabilities" in data:
                print(f"{Colors.GREEN}✓ PASS{Colors.ENDC} | Predict works (Prediction: {data['prediction']})")
            else:
                print(f"{Colors.RED}✗ FAIL{Colors.ENDC} | Missing fields in response")
                all_ok = False
        else:
            print(f"{Colors.RED}✗ FAIL{Colors.ENDC} | Status {response.status_code}")
            all_ok = False
    except Exception as e:
        print(f"{Colors.RED}✗ FAIL{Colors.ENDC} | {str(e)[:50]}")
        all_ok = False
    
    print()
    
    if all_ok:
        print(f"{Colors.GREEN}{Colors.BOLD}✅ All checks passed{Colors.ENDC}")
        return 0
    else:
        print(f"{Colors.RED}{Colors.BOLD}❌ Some checks failed{Colors.ENDC}")
        print(f"\n{Colors.YELLOW}Solución:{Colors.ENDC}")
        print("1. Verifica que 'docker compose up' está corriendo")
        print("2. Espera 30 segundos para que los servicios inicien")
        print("3. Revisa logs: docker compose logs")
        return 1

if __name__ == "__main__":
    sys.exit(main())
