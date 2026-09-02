"""
Integration tests - hit the actual FastAPI app in-process via TestClient,
no real server/network needed.
"""
from fastapi.testclient import TestClient
from app.api.main import app

client = TestClient(app)

VALID_ATTRITION_PAYLOAD = {
    "age": 29,
    "department": "Engineering",
    "work_mode": "Remote",
    "tenure_months": 4,
    "monthly_salary": 55000,
    "months_since_last_hike": 22,
    "productivity_score": 3.2,
    "leaves_last_90_days": 6,
    "manager_change_count": 2,
    "performance_rating": 2,
}

VALID_PROMOTION_PAYLOAD = {
    "department": "Engineering",
    "current_level": "Mid",
    "tenure_in_role_months": 30,
    "avg_performance_rating_last_4q": 4.5,
    "goals_completed_pct": 90.0,
    "peer_review_score": 8.5,
    "training_hours_last_year": 40,
    "cross_team_projects": 2,
    "manager_recommendation": 1,
}


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_attrition_valid_payload():
    response = client.post("/predict/attrition", json=VALID_ATTRITION_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["model_name"] == "attrition"
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0
    assert "request_id" in body


def test_predict_promotion_valid_payload():
    response = client.post("/predict/promotion", json=VALID_PROMOTION_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["model_name"] == "promotion"
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0


def test_predict_attrition_invalid_department_rejected():
    bad_payload = {**VALID_ATTRITION_PAYLOAD, "department": "NotARealDept"}
    response = client.post("/predict/attrition", json=bad_payload)
    assert response.status_code == 422  # Pydantic validation error


def test_predict_attrition_missing_field_rejected():
    bad_payload = {**VALID_ATTRITION_PAYLOAD}
    del bad_payload["age"]
    response = client.post("/predict/attrition", json=bad_payload)
    assert response.status_code == 422


def test_predict_attrition_out_of_range_rejected():
    bad_payload = {**VALID_ATTRITION_PAYLOAD, "performance_rating": 99}
    response = client.post("/predict/attrition", json=bad_payload)
    assert response.status_code == 422


def test_unknown_model_route_returns_404():
    # Directly hitting an undefined model path
    response = client.post("/predict/not-a-real-model", json={})
    assert response.status_code in (404, 405)  # 405 since route doesn't exist at all