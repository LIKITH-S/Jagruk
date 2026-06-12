import os
import json
import tempfile
import pytest
from cv_service.scoring.policy import load_policy, get_default_policy

def test_get_default_policy():
    policy = get_default_policy()
    assert "version" in policy
    assert "scoring_weights" in policy
    assert policy["scoring_weights"]["model_weight"] == 0.7

def test_load_policy_default_when_none():
    policy = load_policy(None)
    assert policy["scoring_weights"]["model_weight"] == 0.7

def test_load_policy_invalid_path():
    policy = load_policy("/path/does/not/exist.json")
    assert policy["scoring_weights"]["model_weight"] == 0.7

def test_load_policy_valid_file():
    custom_policy = {
        "version": "2.0",
        "scoring_weights": {"model_weight": 0.5, "cv_weight": 0.5},
        "thresholds": {},
        "confidence_floor": 0.1,
        "fake_detection_thresholds": {}
    }
    with tempfile.NamedTemporaryFile('w', delete=False, suffix='.json') as f:
        json.dump(custom_policy, f)
        temp_path = f.name
        
    try:
        policy = load_policy(temp_path)
        assert policy["version"] == "2.0"
        assert policy["scoring_weights"]["model_weight"] == 0.5
    finally:
        os.remove(temp_path)

def test_load_policy_invalid_json():
    with tempfile.NamedTemporaryFile('w', delete=False, suffix='.json') as f:
        f.write("{invalid_json:")
        temp_path = f.name
        
    try:
        policy = load_policy(temp_path)
        # Should fallback to default on invalid JSON
        assert policy["version"] == "1.0"
        assert policy["scoring_weights"]["model_weight"] == 0.7
    finally:
        os.remove(temp_path)
