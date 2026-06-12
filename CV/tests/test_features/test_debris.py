import numpy as np
import pytest
from cv_service.features.debris import extract_debris_ratio

def test_extract_debris_ratio_uniform():
    # Uniform matrix (no edges, no variance)
    gray = np.full((100, 100), 100, dtype=np.uint8)
    score = extract_debris_ratio(gray)
    assert score == 0.0

def test_extract_debris_ratio_noise():
    # Random noise matrix (high edges, high variance)
    gray = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    score = extract_debris_ratio(gray)
    assert score > 0.1 # Should detect significant frequency
    assert score <= 1.0 # Should be bounded to 1.0
