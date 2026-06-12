import cv2
import numpy as np

def auto_canny(image: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    """Compute Canny edge detection with dynamic thresholds based on median variance."""
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(image, lower, upper)
    return edges
