import pytest
import cv2
import numpy as np
import os
import tempfile

from cv_service.fake_detection.pipeline import analyze_fake_signals


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


def test_analyze_fake_signals_valid_image(temp_jpeg_image):
    """Pipeline should return ok with both ELA and pHash for a valid JPEG."""
    result = analyze_fake_signals(temp_jpeg_image)
    assert result["status"] == "ok"
    assert result["ela_score"] is not None
    assert isinstance(result["ela_score"], float)
    assert 0.0 <= result["ela_score"] <= 1.0
    assert result["phash"] is not None
    assert isinstance(result["phash"], str)


def test_analyze_fake_signals_numpy_input(synthetic_numpy_image):
    """Pipeline should accept numpy arrays."""
    result = analyze_fake_signals(synthetic_numpy_image)
    assert result["status"] == "ok"
    assert result["ela_score"] is not None


def test_analyze_fake_signals_invalid_path():
    """Pipeline should return error for nonexistent file."""
    result = analyze_fake_signals("/nonexistent.jpg")
    assert result["status"] == "error"
