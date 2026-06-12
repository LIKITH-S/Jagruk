import logging
import os
from fastapi import FastAPI
from cv_service.api.routes import router
from cv_service.scoring.policy import load_policy
from cv_service.models.inference import DamageInference

from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up Jagruk CV API...")
    
    # Load Policy
    policy_path = os.getenv("CV_POLICY_PATH", "cv_policy.json")
    app.state.policy = load_policy(policy_path)
    
    # Load Model
    model_path = os.getenv("CV_MODEL_PATH", "models/best_model.pth")
    if os.path.exists(model_path):
        app.state.model = DamageInference(model_path)
        logger.info(f"Model loaded from {model_path}")
    else:
        logger.error(f"Model file not found at {model_path}")
        # In production, we might want to fail here, but for now we'll log it.
        app.state.model = None
    
    yield

app = FastAPI(title="Jagruk CV API", version="1.0.0", lifespan=lifespan)

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "model_loaded": app.state.model is not None,
        "policy_loaded": app.state.policy is not None
    }

app.include_router(router)
