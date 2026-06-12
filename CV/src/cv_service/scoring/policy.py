import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def get_default_policy() -> dict:
    """Returns the default policy configuration."""
    return {
        "version": "1.0",
        "scoring_weights": {
            "model_weight": 0.7,
            "cv_weight": 0.3
        },
        "thresholds": {
            "debris_ratio": 0.5,
            "broken_edges": 0.5,
            "collapse_features": 0.5
        },
        "confidence_floor": 0.05,
        "fake_detection_thresholds": {
            "ela_suspicion": 0.7,
            "phash_distance": 10
        }
    }

def load_policy(config_path: str = None) -> dict:
    """
    Loads the policy configuration from a JSON file.
    If the file does not exist or config_path is None, falls back to the default policy.
    """
    if config_path:
        path = Path(config_path)
        if path.is_file():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    policy = json.load(f)
                    logger.info(f"Loaded custom policy from {config_path}")
                    return policy
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse policy JSON from {config_path}: {e}")
            except Exception as e:
                logger.error(f"Error loading policy from {config_path}: {e}")
        else:
            logger.warning(f"Policy file not found at {config_path}. Falling back to default.")
            
    logger.info("Using default policy configuration.")
    return get_default_policy()
