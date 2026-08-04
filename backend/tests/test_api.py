from fastapi.testclient import TestClient

def test_health_check(app):
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200

def test_predict_selangor(app):
    client = TestClient(app)
    response = client.post("/predict/selangor", json={"swing": 0.05})
    assert response.status_code == 200
    assert "avg_probability" in response.json()