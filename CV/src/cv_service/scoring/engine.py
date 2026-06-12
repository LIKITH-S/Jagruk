import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

def calculate_final_score(ml_probs: dict, cv_features: dict, policy: dict) -> tuple[float, str]:
    """
    Calculates the final damage score based on ML probabilities and CV features.
    Returns a tuple of (damage_score, damage_label).
    """
    weights = policy.get("scoring_weights", {})
    model_weight = weights.get("model_weight", 0.7)
    cv_weight = weights.get("cv_weight", 0.3)
    
    # Extract ML score (assume it's continuous 0.0 to 1.0)
    ml_score = ml_probs.get("damage_score", 0.0)
    if not isinstance(ml_score, (int, float)):
        ml_score = 0.0
        
    # Extract CV features
    cv_feats = cv_features.get("features", {}) if "features" in cv_features else cv_features
    debris_ratio = cv_feats.get("debris_ratio", 0.0)
    broken_edges = cv_feats.get("broken_edges", 0.0)
    collapse_features = cv_feats.get("collapse_features", 0.0)
    
    # Handle error or missing features by treating them as 0
    cv_values = [v for v in [debris_ratio, broken_edges, collapse_features] if isinstance(v, (int, float))]
    cv_score = sum(cv_values) / len(cv_values) if cv_values else 0.0
    
    # Calculate weighted final score
    final_score = (ml_score * model_weight) + (cv_score * cv_weight)
    final_score = max(0.0, min(1.0, final_score))
    
    # Determine label (no_damage, minor_damage, major_damage, destroyed)
    if final_score < 0.25:
        label = "no_damage"
    elif final_score < 0.5:
        label = "minor_damage"
    elif final_score < 0.75:
        label = "major_damage"
    else:
        label = "destroyed"
        
    return final_score, label

def build_audit_response(report_id: str, damage_score: float, damage_label: str, 
                         confidence: float, fake_image_score: float, 
                         ml_probs: dict, cv_features: dict, 
                         quality_checks: dict, fake_checks: dict,
                         policy_version: str = "cv_policy_v1.0") -> dict:
    """
    Builds the structured JSON response dictionary conforming to Fusion expectations.
    """
    cv_feats = cv_features.get("features", {}) if "features" in cv_features else cv_features
    qc = quality_checks.get("quality_checks", {}) if "quality_checks" in quality_checks else quality_checks
    fc = fake_checks.get("fake_checks", {}) if "fake_checks" in fake_checks else fake_checks

    return {
        "report_id": report_id,
        "damage_score": round(damage_score, 4),
        "damage_label": damage_label,
        "confidence": round(confidence, 4),
        "fake_image_score": round(fake_image_score, 4),
        "explain": {
            "debris_ratio": round(cv_feats.get("debris_ratio", 0.0), 4),
            "broken_edges": round(cv_feats.get("broken_edges", 0.0), 4),
            "collapse_features": round(cv_feats.get("collapse_features", 0.0), 4),
            "model_probability": round(ml_probs.get("model_probability", 0.0), 4),
            "quality_checks": {
                "resolution_px": qc.get("resolution", [0, 0]),
                "blur_score": round(qc.get("blur_score", 0.0), 4)
            },
            "fake_checks": {
                "ela": round(fc.get("ela_score", 0.0), 4),
                "phash_sim": round(fc.get("phash_sim", 0.0), 4)
            }
        },
        "model_version": "cv_v1.0",
        "policy_version": policy_version,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
