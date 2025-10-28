import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def test_health():
    """Test health endpoint"""
    print("🔍 Testing /api/v1/health...")
    response = requests.get(f"{BASE_URL}/api/v1/health")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    assert response.status_code == 200
    print("   ✅ Health check passed\n")

def test_root():
    """Test root endpoint"""
    print("🔍 Testing /...")
    response = requests.get(f"{BASE_URL}/")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    assert response.status_code == 200
    print("   ✅ Root endpoint passed\n")

def test_train():
    """Test train endpoint"""
    print("🔍 Testing /api/v1/train...")
    response = requests.post(f"{BASE_URL}/api/v1/train")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json() if response.status_code == 200 else response.text}")
    print(f"   ℹ️  Check if this is expected behavior\n")

def test_predict():
    """Test predict endpoint"""
    print("🔍 Testing /api/v1/predict...")
    # Ajusta el payload según tu modelo
    payload = {
        "features": [0.1, 0.2, 0.3]  # Ejemplo
    }
    response = requests.post(
        f"{BASE_URL}/api/v1/predict",
        json=payload
    )
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json() if response.status_code == 200 else response.text}")
    print(f"   ℹ️  Check if this is expected behavior\n")

if __name__ == "__main__":
    print("=" * 50)
    print("🚀 API VALIDATION TESTS")
    print("=" * 50 + "\n")
    
    try:
        test_health()
        test_root()
        test_train()
        test_predict()
        print("=" * 50)
        print("✅ All basic tests completed!")
        print("=" * 50)
    except Exception as e:
        print(f"\n❌ Error: {e}")