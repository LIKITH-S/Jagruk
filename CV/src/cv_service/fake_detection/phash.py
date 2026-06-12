import cv2
import numpy as np
import logging

try:
    import imagehash
    from PIL import Image
except ImportError:
    raise ImportError(
        "imagehash and Pillow are required for pHash detection. "
        "Install them with: pip install imagehash Pillow"
    )

logger = logging.getLogger(__name__)

# Constants
PHASH_HASH_SIZE = 8  # 8x8 = 64-bit hash
PHASH_SUSPICION_THRESHOLD = 10  # Hamming distance <= 10 = suspicious


def compute_phash(image_input) -> dict:
    """
    Compute perceptual hash (pHash) for an image.

    Accepts either a file path (str) or a pre-loaded numpy array (BGR format).
    Returns a hex string representation of the 64-bit perceptual hash.

    Args:
        image_input: File path (str) or numpy array (BGR format).

    Returns:
        dict: {"status": "ok", "phash": str} on success,
              {"status": "error", "reason": str} on failure.
    """
    try:
        if isinstance(image_input, str):
            img = Image.open(image_input)
        elif isinstance(image_input, np.ndarray):
            # Convert BGR (OpenCV) to RGB (PIL)
            rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
        else:
            logger.error(f"Unsupported image_input type: {type(image_input)}")
            return {"status": "error", "reason": "unsupported_type"}

        hash_value = imagehash.phash(img, hash_size=PHASH_HASH_SIZE)

        return {"status": "ok", "phash": str(hash_value)}

    except Exception as e:
        logger.error(f"Exception during pHash computation: {e}")
        return {"status": "error", "reason": str(e)}


def compute_hamming_distance(hash1: str, hash2: str) -> dict:
    """
    Compute Hamming distance between two perceptual hash hex strings.

    Args:
        hash1: First pHash hex string.
        hash2: Second pHash hex string.

    Returns:
        dict: {"status": "ok", "hamming_distance": int, "is_suspicious": bool,
               "threshold": int} on success,
              {"status": "error", "reason": str} on failure.
    """
    try:
        h1 = imagehash.hex_to_hash(hash1)
        h2 = imagehash.hex_to_hash(hash2)

        distance = h1 - h2
        is_suspicious = distance <= PHASH_SUSPICION_THRESHOLD

        return {
            "status": "ok",
            "hamming_distance": int(distance),
            "is_suspicious": is_suspicious,
            "threshold": PHASH_SUSPICION_THRESHOLD,
        }

    except Exception as e:
        logger.error(f"Exception during Hamming distance computation: {e}")
        return {"status": "error", "reason": str(e)}
