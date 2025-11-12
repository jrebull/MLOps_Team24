import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture(scope="session")
def client():
    """Proporciona TestClient para todos los tests"""
    return TestClient(app)
