from pydantic import BaseModel
from typing import Optional, Any, Dict

class AnalyzeRequest(BaseModel):
    report_id: Optional[str] = None
    image_base64: str

class QualityChecks(BaseModel):
    resolution_px: list
    blur_score: float

class FakeChecks(BaseModel):
    ela: float
    phash_sim: float

class Explain(BaseModel):
    debris_ratio: float
    broken_edges: float
    collapse_features: float
    model_probability: float
    quality_checks: QualityChecks
    fake_checks: FakeChecks

class AnalyzeResponse(BaseModel):
    report_id: str
    damage_score: float
    damage_label: str
    confidence: float
    fake_image_score: float
    explain: Explain
    model_version: str
    policy_version: str
    timestamp: str
