import pytest

from cv_service.quality.trust import adjust_confidence


def test_adjust_confidence_no_penalty():
    """No penalty when blur and resolution are above thresholds."""
    result = adjust_confidence(0.85, blur_score=500.0, resolution_px=(640, 480))
    assert result["final_confidence"] == 0.85
    assert result["quality_breakdown"]["blur_factor"] == 1.0
    assert result["quality_breakdown"]["resolution_factor"] == 1.0
    assert result["quality_breakdown"]["confidence_floor_applied"] == False


def test_adjust_confidence_blur_penalty():
    """Blur below threshold should reduce confidence."""
    result = adjust_confidence(0.80, blur_score=50.0, resolution_px=(640, 480))
    # blur_factor = 50/100 = 0.5, resolution_factor = 1.0
    # confidence = 0.80 * 0.5 * 1.0 = 0.4
    assert result["final_confidence"] == 0.4
    assert result["quality_breakdown"]["blur_factor"] == 0.5


def test_adjust_confidence_resolution_penalty():
    """Resolution below threshold should reduce confidence."""
    result = adjust_confidence(0.80, blur_score=200.0, resolution_px=(112, 112))
    # blur_factor = 1.0, resolution_factor = 112/224 = 0.5
    # confidence = 0.80 * 1.0 * 0.5 = 0.4
    assert result["final_confidence"] == 0.4
    assert result["quality_breakdown"]["resolution_factor"] == 0.5


def test_adjust_confidence_floor_applied():
    """Very low quality should be floored at confidence_floor."""
    result = adjust_confidence(0.01, blur_score=10.0, resolution_px=(50, 50))
    # blur_factor = 10/100 = 0.1, resolution_factor = 50/224 ≈ 0.223
    # raw = 0.01 * 0.1 * 0.223 ≈ 0.000223 → floored to 0.05
    assert result["final_confidence"] == 0.05
    assert result["quality_breakdown"]["confidence_floor_applied"] == True


def test_adjust_confidence_combined_penalty():
    """Combined blur + resolution penalty."""
    result = adjust_confidence(0.90, blur_score=50.0, resolution_px=(112, 112))
    # blur_factor = 0.5, resolution_factor = 0.5
    # confidence = 0.90 * 0.5 * 0.5 = 0.225
    assert result["final_confidence"] == 0.225
