import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Constants
ELA_JPEG_QUALITY = 90
ELA_MAX_DIFF = 255.0


def compute_ela_score(image_input) -> dict:
    """
    Compute Error Level Analysis (ELA) score for manipulation detection.

    Accepts either a file path (str) or a pre-loaded numpy array.
    Re-compresses the image at JPEG quality=90 in-memory, computes the
    absolute pixel difference, and returns a normalized [0,1] suspicion score.

    Args:
        image_input: File path (str) or numpy array (BGR format).

    Returns:
        dict: {"status": "ok", "ela_score": float} on success,
              {"status": "error", "reason": str} on failure.
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
            logger.error(f"Unsupported image_input type: {type(image_input)}")
            return {"status": "error", "reason": "unsupported_type"}

        # Re-compress in-memory at JPEG quality=90 (no temp file)
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), ELA_JPEG_QUALITY]
        success, buffer = cv2.imencode('.jpg', image, encode_params)

        if not success:
            logger.error("JPEG encoding failed during ELA computation")
            return {"status": "error", "reason": "jpeg_encode_failed"}

        # Decode the buffer back to a numpy array
        recompressed = cv2.imdecode(
            np.frombuffer(buffer, dtype=np.uint8), cv2.IMREAD_COLOR
        )

        # Compute absolute pixel difference
        diff = cv2.absdiff(image, recompressed)

        # Normalize to [0, 1] using mean across all channels
        ela_score = float(np.mean(diff) / ELA_MAX_DIFF)

        return {"status": "ok", "ela_score": ela_score}

    except Exception as e:
        logger.error(f"Exception during ELA computation: {e}")
        return {"status": "error", "reason": "exception"}
