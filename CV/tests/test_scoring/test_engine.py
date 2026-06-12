import pytest
from cv_service.scoring.engine import calculate_final_score, build_audit_response
from cv_service.scoring.policy import get_default_policy

def test_calculate_final_score_normal():
    policy = get_default_policy()
    ml_probs = {"damage_score": 0.8}
    cv_features = {
        "features": {
            "debris_ratio": 0.6,
            "broken_edges": 0.4,
            "collapse_features": 0.5
        }
    }
    
    score, label = calculate_final_score(ml_probs, cv_features, policy)
    # ML: 0.8 * 0.7 = 0.56
    # CV: (0.6+0.4+0.5)/3 = 0.5; 0.5 * 0.3 = 0.15
    # Total = 0.71
    assert abs(score - 0.71) < 1e-5
    assert label == "major_damage"

def test_calculate_final_score_missing_cv_features():
    policy = get_default_policy()
    ml_probs = {"damage_score": 0.8}
    cv_features = {} # Empty
    
    score, label = calculate_final_score(ml_probs, cv_features, policy)
    # CV score is 0.0 because no valid features
    # Total = 0.8 * 0.7 = 0.56
    assert abs(score - 0.56) < 1e-5
    assert label == "major_damage"

def test_calculate_final_score_invalid_ml_score():
    policy = get_default_policy()
    ml_probs = {"damage_score": "invalid"}
    cv_features = {
        "debris_ratio": 1.0,
        "broken_edges": 1.0,
        "collapse_features": 1.0
    }
    
    score, label = calculate_final_score(ml_probs, cv_features, policy)
    # ML score is 0.0 due to invalid string
    # CV score is 1.0 * 0.3 = 0.3
    # Total = 0.3
    assert abs(score - 0.3) < 1e-5
    assert label == "minor_damage"

def test_calculate_final_score_bounds():
    policy = get_default_policy()
    policy["scoring_weights"] = {"model_weight": 2.0, "cv_weight": 2.0} # Ensure it goes over 1.0
    ml_probs = {"damage_score": 1.0}
    cv_features = {"debris_ratio": 1.0}
    
    score, label = calculate_final_score(ml_probs, cv_features, policy)
    assert score == 1.0 # Clamped
    assert label == "destroyed"

def test_build_audit_response():
    resp = build_audit_response(
        report_id="123",
        damage_score=0.71,
        damage_label="major_damage",
        confidence=0.85,
        fake_image_score=0.1,
        ml_probs={"model_probability": 0.8},
        cv_features={"debris_ratio": 0.6},
        quality_checks={"blur_score": 100, "resolution": [224, 224]},
        fake_checks={"ela_score": 0.1, "phash_sim": 0.9}
    )
    
    assert resp["report_id"] == "123"
    assert resp["damage_score"] == 0.71
    assert resp["damage_label"] == "major_damage"
    assert "explain" in resp
    assert resp["explain"]["debris_ratio"] == 0.6
    assert resp["explain"]["quality_checks"]["blur_score"] == 100
    assert resp["explain"]["fake_checks"]["ela"] == 0.1
    assert resp["explain"]["model_probability"] == 0.8
