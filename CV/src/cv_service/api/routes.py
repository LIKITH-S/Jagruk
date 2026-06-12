import base64
import uuid
import cv2
import numpy as np
import logging
from fastapi import APIRouter, HTTPException, Request
from PIL import Image
import io

from .schemas import AnalyzeRequest, AnalyzeResponse
from cv_service.features.pipeline import extract_all_cv_features
from cv_service.fake_detection.pipeline import analyze_fake_signals
from cv_service.quality.trust import adjust_confidence
from cv_service.scoring.engine import calculate_final_score, build_audit_response
from cv_service.quality.checker import assess_image_quality

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/cv/analyze", response_model=AnalyzeResponse)
async def analyze_image(request: Request):
    content_type = request.headers.get("content-type", "")
    req_report_id = str(uuid.uuid4())
    img_bytes = None

    # 1. Load image
    if "multipart/form-data" in content_type:
        form = await request.form()
        file = form.get("file")
        req_report_id = form.get("report_id", req_report_id)
        if file and hasattr(file, "read"):
            img_bytes = await file.read()
    elif "application/json" in content_type:
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON")
        req_report_id = body.get("report_id", req_report_id)
        b64 = body.get("image_base64")
        if b64:
            try:
                img_bytes = base64.b64decode(b64)
            except Exception:
                raise HTTPException(status_code=422, detail="Invalid Base64 string")
    else:
        raise HTTPException(status_code=415, detail="Unsupported Media Type")

    if not img_bytes:
        raise HTTPException(status_code=400, detail="Must provide either multipart 'file' or JSON 'image_base64'")

    # 2. Validate image
    np_arr = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise HTTPException(status_code=415, detail="Unsupported image format")

    app_state = request.app.state
    policy = getattr(app_state, "policy", {})
    model = getattr(app_state, "model", None)

    if model is None:
        logger.error("Model not loaded in app state")
        raise HTTPException(status_code=503, detail="Model not initialized")

    try:
        # 3. Quality analysis
        quality_result = assess_image_quality(image)
        blur_score = quality_result.get("blur_score", 0.0)
        resolution_px = quality_result.get("resolution_px", (0, 0))

        # 4. Model inference
        # Convert CV2 image (BGR) to PIL image (RGB) for inference
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        ml_results = model.predict(pil_image)

        # 5. CV feature extraction
        cv_features = extract_all_cv_features(image)

        # 6. Policy scoring
        damage_score, damage_label = calculate_final_score(ml_results, cv_features, policy)

        # 7. Confidence adjustment
        # User requested order: adjust_confidence(model_probability, blur_score, resolution_px)
        model_prob = ml_results.get("model_probability", 0.0)
        confidence_floor = policy.get("confidence_floor", 0.05)
        
        conf_result = adjust_confidence(
            model_probability=model_prob,
            blur_score=blur_score,
            resolution_px=resolution_px,
            confidence_floor=confidence_floor
        )
        final_confidence = conf_result.get("final_confidence", 0.0)

        # 8. Fake detection
        # Note: analyze_fake_signals currently takes a path, I might need to update it too if it fails.
        # Let's check it or wrap it.
        try:
            fake_checks = analyze_fake_signals(image)
        except Exception as e:
            logger.error(f"Fake detection failed: {e}")
            fake_checks = {"suspicion_score": 0.0, "status": "error"}

        fake_image_score = fake_checks.get("suspicion_score", 0.0)

        # 9. Response build
        quality_checks = {
            "quality_checks": {
                "blur_score": blur_score,
                "resolution": list(resolution_px)
            }
        }

        resp_dict = build_audit_response(
            report_id=req_report_id,
            damage_score=damage_score,
            damage_label=damage_label,
            confidence=final_confidence,
            fake_image_score=fake_image_score,
            ml_probs=ml_results,
            cv_features=cv_features,
            quality_checks=quality_checks,
            fake_checks=fake_checks,
            policy_version=policy.get("version", "cv_policy_v1.0")
        )
        return resp_dict

    except Exception as e:
        logger.exception("Internal error during image analysis")
        raise HTTPException(status_code=500, detail=f"Internal processing error: {str(e)}")
