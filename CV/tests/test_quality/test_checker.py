import os
import tempfile
import cv2
import numpy as np
import pytest

from cv_service.quality.checker import assess_image_quality

def test_assess_image_quality_unreadable():
    # Test with non-existent file
    result = assess_image_quality("non_existent_file.png")
    assert result["status"] == "error"
    assert result["reason"] == "unreadable"

def test_assess_image_quality_valid():
    # Create a temporary random image
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        # Create a 100x100 RGB image with random noise
        # This will be blurry but readable
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(tmp_path, img)

        result = assess_image_quality(tmp_path)
        
        assert result["status"] == "ok"
        assert result["resolution_px"] == (100, 100)
        assert "blur_score" in result
        assert isinstance(result["blur_score"], float)
    
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
