"""Defines the API routes"""

from fastapi import APIRouter
from src.ml_project_template.schemas import ModelRequest, ModelResponse
from src.ml_project_template.api.inference import predict

router = APIRouter(prefix="/api", tags=["inference"])


# Include whatever routers you need here
@router.post("/predict", response_model=ModelResponse)
def run_inference(request: ModelRequest):
    """Performs inference using the loaded model"""
    return {"prediction": predict(request.request_input)}
