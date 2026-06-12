import cv2
import numpy as np
from cv_service.utils.image_ops import auto_canny

def extract_broken_edges(gray_image: np.ndarray, edges: np.ndarray = None) -> float:
    """
    Computes a broken edge metric [0.0, 1.0].
    1.0 means fully broken/no long geometric lines (damaged).
    0.0 means perfectly long structure.
    """
    if gray_image is None or gray_image.size == 0:
        return 1.0

    if edges is None:
        edges = auto_canny(gray_image)

    # Probabilistic Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, minLineLength=10, maxLineGap=5)
    
    if lines is None or len(lines) == 0:
        return 1.0

    total_length = 0.0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        total_length += length
        
    avg_length = total_length / len(lines)
    
    h, w = gray_image.shape[:2]
    max_theoretical_length = np.sqrt(w**2 + h**2)
    
    # Fully broken logic: high avg_length means very structured, so score closer to 0
    # No lines means fully unstructured, so score exactly 1
    # Thus: 1.0 - (avg_length / max)
    broken_score = 1.0 - (avg_length / max_theoretical_length)
    
    return float(min(1.0, max(0.0, broken_score)))
