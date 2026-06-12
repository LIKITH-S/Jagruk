import logging

from cv_service.quality.checker import assess_image_quality
from cv_service.fake_detection.pipeline import analyze_fake_signals

logger = logging.getLogger(__name__)

# Constants
DEFAULT_CONFIDENCE_FLOOR = 0.05
BLUR_THRESHOLD = 100.0  # Laplacian variance below this = quality penalty
RESOLUTION_THRESHOLD = 224  # Minimum dimension in pixels (matches model input size)


def adjust_confidence(
    model_probability: float,
    blur_score: float,
    resolution_px: tuple,
    confidence_floor: float = DEFAULT_CONFIDENCE_FLOOR,
) -> dict:
    """
    Apply multiplicative confidence penalties based on image quality.

    Uses blur and resolution factors to penalize model confidence when
    input quality is degraded. A configurable floor prevents zero-confidence
    outputs that would break downstream probability reasoning.

    Args:
        model_probability: Raw model output probability [0, 1].
        blur_score: Laplacian variance from quality checker (higher = sharper).
        resolution_px: Tuple of (width, height) in pixels.
        confidence_floor: Minimum confidence value (default 0.05).

    Returns:
        dict: {"final_confidence": float, "quality_breakdown": {...}}
    """
    # Blur penalty: penalize very blurry images (below threshold)
    blur_factor = min(1.0, blur_score / BLUR_THRESHOLD)

    # Resolution penalty: penalize very small images (below model input size)
    min_dim = min(resolution_px[0], resolution_px[1])
    resolution_factor = min(1.0, min_dim / RESOLUTION_THRESHOLD)

    # Multiplicative adjustment
    confidence = model_probability * blur_factor * resolution_factor

    # Apply confidence floor
    confidence = max(confidence_floor, confidence)

    return {
        "final_confidence": round(confidence, 6),
        "quality_breakdown": {
            "blur_factor": round(blur_factor, 6),
            "resolution_factor": round(resolution_factor, 6),
            "confidence_floor_applied": confidence == confidence_floor,
        },
    }


def compute_trust_signals(
    image_path: str,
    model_probability: float,
    confidence_floor: float = DEFAULT_CONFIDENCE_FLOOR,
) -> dict:
    """
    Aggregate all trust signals for an image into a unified result.

    Combines quality assessment (blur, resolution), fake detection signals
    (ELA, pHash), and confidence adjustment into a single structured output
    ready for consumption by the Phase 5 scoring engine.

    Args:
        image_path: Path to the image file.
        model_probability: Raw model output probability [0, 1].
        confidence_floor: Minimum confidence value (default 0.05).

    Returns:
        dict: Unified trust signal result including final_confidence,
              quality_checks, fake_signals, and floor application status.
    """
    try:
        # Step 1: Assess image quality
        quality_result = assess_image_quality(image_path)
        if quality_result.get("status") != "ok":
            logger.warning(f"Quality check failed for {image_path}")
            return {"status": "error", "reason": "quality_check_failed"}

        blur_score = quality_result["blur_score"]
        resolution_px = quality_result["resolution_px"]

        # Step 2: Run fake detection signals
        fake_result = analyze_fake_signals(image_path)

        # Step 3: Adjust confidence
        confidence_result = adjust_confidence(
            model_probability, blur_score, resolution_px, confidence_floor
        )

        return {
            "status": "ok",
            "final_confidence": confidence_result["final_confidence"],
            "quality_checks": {
                "blur_score": blur_score,
                "resolution_px": resolution_px,
                "blur_factor": confidence_result["quality_breakdown"]["blur_factor"],
                "resolution_factor": confidence_result["quality_breakdown"]["resolution_factor"],
            },
            "fake_signals": fake_result,
            "confidence_floor_applied": confidence_result["quality_breakdown"]["confidence_floor_applied"],
        }

    except Exception as e:
        logger.error(f"Exception in compute_trust_signals: {e}")
        return {"status": "error", "reason": str(e)}
