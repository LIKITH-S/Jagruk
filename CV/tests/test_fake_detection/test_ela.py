import pytest
import cv2
import numpy as np
import os
import tempfile

from cv_service.fake_detection.ela import compute_ela_score


@pytest.fixture
def temp_jpeg_image():
    """Create a temporary JPEG image file for testing."""
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    fd, path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    cv2.imwrite(path, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    yield path
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def synthetic_numpy_image():
    """Create a synthetic BGR numpy array for testing."""
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)


def test_ela_returns_ok_for_valid_image(temp_jpeg_image):
    """ELA should return ok status and a valid score for a real JPEG file."""
    result = compute_ela_score(temp_jpeg_image)
    assert result["status"] == "ok"
    assert 0.0 <= result["ela_score"] <= 1.0


def test_ela_returns_ok_for_numpy_input(synthetic_numpy_image):
    """ELA should accept numpy arrays directly."""
    result = compute_ela_score(synthetic_numpy_image)
    assert result["status"] == "ok"
    assert 0.0 <= result["ela_score"] <= 1.0


def test_ela_returns_error_for_invalid_path():
    """ELA should return error status for a nonexistent file."""
    result = compute_ela_score("/nonexistent/image.jpg")
    assert result["status"] == "error"


def test_ela_score_is_normalized(synthetic_numpy_image):
    """ELA score must always be in [0, 1] range."""
    result = compute_ela_score(synthetic_numpy_image)
    assert result["ela_score"] <= 1.0
    assert result["ela_score"] >= 0.0
