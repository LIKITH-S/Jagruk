import pytest
import cv2
import numpy as np
import os
import tempfile

from cv_service.fake_detection.phash import compute_phash, compute_hamming_distance


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


def test_phash_returns_ok_for_valid_image(temp_jpeg_image):
    """pHash should return ok status and a hex hash string for a JPEG file."""
    result = compute_phash(temp_jpeg_image)
    assert result["status"] == "ok"
    assert isinstance(result["phash"], str)
    assert len(result["phash"]) == 16  # 64-bit hash = 16 hex chars


def test_phash_returns_ok_for_numpy_input(synthetic_numpy_image):
    """pHash should accept numpy arrays directly."""
    result = compute_phash(synthetic_numpy_image)
    assert result["status"] == "ok"
    assert isinstance(result["phash"], str)


def test_phash_identical_images_zero_distance(temp_jpeg_image):
    """Identical images should have Hamming distance of 0."""
    result1 = compute_phash(temp_jpeg_image)
    result2 = compute_phash(temp_jpeg_image)
    assert result1["status"] == "ok"
    assert result2["status"] == "ok"

    dist = compute_hamming_distance(result1["phash"], result2["phash"])
    assert dist["status"] == "ok"
    assert dist["hamming_distance"] == 0
    assert dist["is_suspicious"] == True


def test_phash_different_images_high_distance():
    """Very different images should have high Hamming distance."""
    # All white image
    white = np.ones((100, 100, 3), dtype=np.uint8) * 255
    # Random noise image
    np.random.seed(42)
    noise = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    result1 = compute_phash(white)
    result2 = compute_phash(noise)
    assert result1["status"] == "ok"
    assert result2["status"] == "ok"

    dist = compute_hamming_distance(result1["phash"], result2["phash"])
    assert dist["status"] == "ok"
    assert dist["hamming_distance"] > 10
    assert dist["is_suspicious"] == False


def test_phash_returns_error_for_invalid_path():
    """pHash should return error for nonexistent file."""
    result = compute_phash("/nonexistent.jpg")
    assert result["status"] == "error"
