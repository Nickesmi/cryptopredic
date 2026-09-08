from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List, Optional, Any, Dict
from pydantic import BaseModel

from src.evaluation.prediction_store import PredictionStore
from src.config.settings import PREDICTION_DB_PATH

router = APIRouter(tags=["Evaluation & Tracking"])
store = PredictionStore(PREDICTION_DB_PATH)

@router.get("/predictions")
async def get_predictions(limit: int = 100):
    """Fetch recent evaluated predictions"""
    rows = store.evaluated_rows(limit=limit)
    return {"predictions": rows}

@router.get("/predictions/due")
async def get_due_predictions():
    """Fetch predictions that have expired and need evaluation"""
    rows = store.due_predictions()
    return {"due": rows}

@router.get("/predictions/{prediction_id}")
async def get_prediction(prediction_id: str):
    """Get a specific prediction and its result"""
    pred = store.get_prediction(prediction_id)
    if not pred:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return pred

@router.get("/models/leaderboard")
async def get_leaderboard():
    """Get the current model leaderboard"""
    board = store.leaderboard()
    return {"leaderboard": board}

@router.get("/performance/dashboard")
async def get_performance_dashboard():
    """Get the full dashboard data with rolling accuracy and drift metrics"""
    data = store.dashboard()
    return data

@router.get("/models/{model_id}/drift")
async def check_model_drift(model_id: str, threshold: float = 0.50):
    """Check if a model has drifted below accuracy threshold"""
    # Simple drift detection using leaderboard or evaluated rows
    board = store.leaderboard()
    for row in board:
        if row["model"] == model_id:
            if row.get("directional_accuracy", 1.0) < threshold:
                return {"model_id": model_id, "drift_detected": True, "reason": "Accuracy below threshold"}
    return {"model_id": model_id, "drift_detected": False, "reason": "Performance stable or model unknown"}
