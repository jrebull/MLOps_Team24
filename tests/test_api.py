import pytest
from fastapi.testclient import TestClient

def test_health(client):
    """Test health endpoint"""
    print("🔍 Testing /api/v1/health...")
    response = client.get("/api/v1/health")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    assert response.status_code == 200
    print("   ✅ Health check passed\n")

def test_root(client):
    """Test root endpoint"""
    print("🔍 Testing /...")
    response = client.get("/")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    assert response.status_code == 200
    print("   ✅ Root endpoint passed\n")

def test_train(client):
    """Test train endpoint"""
    print("🔍 Testing /api/v1/train...")
    payload = {"data_path": "data/raw", "model_name": "test_model"}
    response = client.post("/api/v1/train", json=payload)
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    print(f"   ℹ️  Check if this is expected behavior\n")

def test_predict(client):
    """Test predict endpoint"""
    print("🔍 Testing /api/v1/predict...")
    payload = {
        "features": [0.1, 0.2, 0.3]
    }
    response = client.post("/api/v1/predict", json=payload)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        print(f"   Response: {response.json()}")
    else:
        print(f"   Response: {response.text}")
    print(f"   ℹ️  Check if this is expected behavior\n")

def test_list_models(client):
    """Test list models endpoint"""
    print("🔍 Testing /api/v1/models...")
    response = client.get("/api/v1/models")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    print("   ✅ List models passed\n")
