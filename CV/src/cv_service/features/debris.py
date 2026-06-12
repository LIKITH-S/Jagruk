import cv2
import numpy as np
from cv_service.utils.image_ops import auto_canny

def extract_debris_ratio(gray_image: np.ndarray, edges: np.ndarray = None) -> float:
    """
    Extracts a normalized debris score [0.0, 1.0] from a grayscale image using
    edge density and local variance.
    """
    if gray_image is None or gray_image.size == 0:
        return 0.0

    if edges is None:
        edges = auto_canny(gray_image)

    # Edge density
    edge_density = np.count_nonzero(edges) / max(1, edges.size)

    # LAP variance mapped up to a cap
    variance = cv2.Laplacian(gray_image, cv2.CV_64F).var()
    normalized_var = min(1.0, float(variance) / 500.0)

    # Hybrid score
    score = (edge_density * 0.7) + (normalized_var * 0.3)
    
    return float(min(1.0, max(0.0, score)))
