import cv2
import numpy as np
from cv_service.features.pipeline import extract_all_cv_features

def test_extract_all_cv_features_array():
    # Use array
    img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    res = extract_all_cv_features(img)
    
    assert res["status"] == "ok"
    assert "debris_ratio" in res
    assert "broken_edges" in res
    assert "collapse_features" in res

def test_extract_all_cv_features_invalid_path():
    res = extract_all_cv_features("some/invalid/path.png")
    assert res["status"] == "error"
    assert "unreadable" in res["reason"]
