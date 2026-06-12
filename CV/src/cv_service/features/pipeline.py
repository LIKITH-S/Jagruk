import cv2
import numpy as np

from cv_service.utils.image_ops import auto_canny
from cv_service.features.debris import extract_debris_ratio
from cv_service.features.edges import extract_broken_edges
from cv_service.features.collapse import extract_collapse_features

def extract_all_cv_features(image_input) -> dict:
    """
    Computes all three classical CV features optimally. 
    Accepts either an image path (str) or a pre-loaded numpy array.
    """
    if isinstance(image_input, str):
        try:
            image = cv2.imread(image_input)
            if image is None:
                return {"status": "error", "reason": f"unreadable image at {image_input}"}
        except Exception as e:
            return {"status": "error", "reason": f"exception loading image: {str(e)}"}
    else:
        # Assume numpy array
        image = image_input

    # Optimal sharing pipeline
    # Grayscale exactly once
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    # Auto-Canny exactly once
    edges = auto_canny(gray)
    
    debris = extract_debris_ratio(gray, edges)
    broken = extract_broken_edges(gray, edges)
    collapse = extract_collapse_features(gray)
    
    return {
        "status": "ok",
        "debris_ratio": debris,
        "broken_edges": broken,
        "collapse_features": collapse
    }
