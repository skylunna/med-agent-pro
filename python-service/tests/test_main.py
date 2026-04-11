from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_check():
    
    assert app.title == "Python AI Engine"