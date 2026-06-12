import io
import cv2
import numpy as np
import base64
import pytest
from fastapi.testclient import TestClient

# Mock out cv_service features and fake_detection imports before importing main
import sys
from unittest.mock import MagicMock

# Create a small valid test image (e.g., 10x10 black square)
def create_test_image_bytes():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    _, buffer = cv2.imencode('.png', img)
    return buffer.tobytes()

from cv_service.api.main import app

client = TestClient(app)

@pytest.fixture(autouse=True)
def load_app_policy():
    # Simulate startup event manually since TestClient doesn't always trigger it correctly for app.state in some setups
    from cv_service.scoring.policy import load_policy
    app.state.policy = load_policy()
    
    mock_model = MagicMock()
    mock_model.predict.return_value = {
        "damage_score": 0.8,
        "model_probability": 0.9,
        "probs": [0.1, 0.1, 0.8, 0.0]
    }
    app.state.model = mock_model

def test_analyze_no_input():
    response = client.post("/cv/analyze")
    assert response.status_code == 415
    assert "Unsupported Media Type" in response.json()["detail"]

def test_analyze_invalid_image():
    # Sending invalid bytes as an image
    files = {"file": ("test.txt", b"invalid_image_data", "text/plain")}
    response = client.post("/cv/analyze", files=files)
    assert response.status_code == 415

def test_analyze_valid_multipart():
    img_bytes = create_test_image_bytes()
    files = {"file": ("test.png", img_bytes, "image/png")}
    response = client.post("/cv/analyze", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert "report_id" in data
    assert "damage_score" in data
    assert "explain" in data

def test_analyze_valid_base64():
    img_bytes = create_test_image_bytes()
    b64_str = base64.b64encode(img_bytes).decode('utf-8')
    
    payload = {
        "report_id": "test-123",
        "image_base64": b64_str
    }
    response = client.post("/cv/analyze", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["report_id"] == "test-123"
    assert "damage_score" in data
    assert "explain" in data
    assert "model_probability" in data["explain"]

def test_analyze_invalid_base64():
    payload = {
        "report_id": "test-123",
        "image_base64": "invalid_base_64_string_!!!!"
    }
    response = client.post("/cv/analyze", json=payload)
    
    # Depending on how strict the base64 parser is, it might be 422 or 415.
    # Our route catches base64 exception and throws 422.
    assert response.status_code in [422, 415]
