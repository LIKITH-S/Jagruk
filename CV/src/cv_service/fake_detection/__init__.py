"""Fake detection and trust signal modules."""
from cv_service.fake_detection.ela import compute_ela_score
from cv_service.fake_detection.phash import compute_phash, compute_hamming_distance
from cv_service.fake_detection.pipeline import analyze_fake_signals
