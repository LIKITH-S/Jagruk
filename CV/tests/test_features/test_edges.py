import cv2
import numpy as np
from cv_service.features.edges import extract_broken_edges

def test_extract_broken_edges_no_lines():
    # Uniform matrix (no edges)
    gray = np.full((100, 100), 100, dtype=np.uint8)
    score = extract_broken_edges(gray)
    assert score == 1.0 # 1.0 = fully unstructured/broken

def test_extract_broken_edges_with_lines():
    # Create black image
    gray = np.zeros((100, 100), dtype=np.uint8)
    # Draw a long bright straight line matching perfectly structured elements
    cv2.line(gray, (10, 10), (90, 90), 255, 2)
    
    score = extract_broken_edges(gray)
    assert score < 1.0 # Should have a significantly lower score because line is long
