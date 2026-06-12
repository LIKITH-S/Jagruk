import cv2
import numpy as np

def extract_collapse_features(gray_image: np.ndarray) -> float:
    """
    Computes a collapse feature metric [0.0, 1.0].
    Uses Shi-Tomasi corner detection to identify cluster density indicative
    of debris pile bounds and crumbled geography.
    """
    if gray_image is None or gray_image.size == 0:
        return 0.0

    # Max corners heuristically tuned for standard 224x224 crops of buildings
    MAX_CORNERS = 500.0

    corners = cv2.goodFeaturesToTrack(
        gray_image, 
        maxCorners=1000, 
        qualityLevel=0.01, 
        minDistance=5
    )

    if corners is None:
        return 0.0

    num_corners = len(corners)
    
    # Normalize with standard fallback
    normalized_score = min(1.0, num_corners / MAX_CORNERS)

    return float(normalized_score)
