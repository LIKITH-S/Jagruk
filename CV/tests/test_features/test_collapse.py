import cv2
import numpy as np
from cv_service.features.collapse import extract_collapse_features

def test_extract_collapse_features_no_corners():
    # Uniform matrix (no corners)
    gray = np.full((100, 100), 100, dtype=np.uint8)
    score = extract_collapse_features(gray)
    assert score == 0.0

def test_extract_collapse_features_high_corners():
    # Checkerboard pattern with 20x20 blocks to produce distinct corners
    gray = np.zeros((100, 100), dtype=np.uint8)
    for i in range(0, 100, 20):
        for j in range(0, 100, 20):
            if (i // 20 + j // 20) % 2 == 0:
                gray[i:i+20, j:j+20] = 255
    
    score = extract_collapse_features(gray)
    assert score > 0.02 # Should detect a significant density of corners
    assert score <= 1.0 # Bounded maximum
