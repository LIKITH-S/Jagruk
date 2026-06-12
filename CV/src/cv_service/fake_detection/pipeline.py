import cv2
import logging

from cv_service.fake_detection.ela import compute_ela_score
from cv_service.fake_detection.phash import compute_phash

logger = logging.getLogger(__name__)


def analyze_fake_signals(image_input) -> dict:
    """
    Run all fake detection signals on an image.

    Aggregates ELA (manipulation detection) and pHash (duplicate detection)
    into a single result dictionary. Partial results are acceptable —
    if one signal fails, the other is still returned.

    Accepts either a file path (str) or a pre-loaded numpy array (BGR format).

    Args:
        image_input: File path (str) or numpy array (BGR format).

    Returns:
        dict: Aggregated fake detection signals with status, ela_score,
              phash, and any error details.
    """
    try:
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            if image is None:
                logger.error(f"Failed to read image at {image_input}")
                return {"status": "error", "reason": "unreadable"}
        else:
            image = image_input

        # Run ELA on numpy array
        ela_result = compute_ela_score(image)

        # Run pHash — pass original input for best compatibility
        # (str paths work directly with PIL, numpy arrays are converted inside)
        phash_result = compute_phash(image_input)

        # Log warnings for individual failures (but don't fail overall)
        if ela_result.get("status") == "error":
            logger.warning(f"ELA computation failed: {ela_result.get('reason')}")
        if phash_result.get("status") == "error":
            logger.warning(f"pHash computation failed: {phash_result.get('reason')}")

        return {
            "status": "ok",
            "ela_score": ela_result.get("ela_score", None),
            "phash": phash_result.get("phash", None),
            "ela_error": ela_result.get("reason") if ela_result.get("status") == "error" else None,
            "phash_error": phash_result.get("reason") if phash_result.get("status") == "error" else None,
        }

    except Exception as e:
        logger.error(f"Exception in analyze_fake_signals: {e}")
        return {"status": "error", "reason": str(e)}
