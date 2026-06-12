import cv2
import logging
import numpy as np

logger = logging.getLogger(__name__)

def assess_image_quality(image_input) -> dict:
    """
    Assess resolution and blur of an image.
    Accepts either a file path (str) or a numpy array (image).
    """
    try:
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            if image is None:
                logger.error(f"Failed to read image at {image_input}")
                return {"status": "error", "reason": "unreadable"}
        elif isinstance(image_input, np.ndarray):
            image = image_input
        else:
            logger.error(f"Unsupported image input type: {type(image_input)}")
            return {"status": "error", "reason": "unsupported_input"}

        h, w = image.shape[:2]
        # Calculate Laplacian variance for blur detection
        # Ensure image is grayscale for Laplacian if needed, but cv2.Laplacian works on color too
        blur_score = cv2.Laplacian(image, cv2.CV_64F).var()

        return {
            "status": "ok",
            "resolution_px": (w, h),
            "blur_score": float(blur_score)
        }
    except Exception as e:
        logger.error(f"Exception while assessing image quality: {e}")
        return {"status": "error", "reason": "exception"}
