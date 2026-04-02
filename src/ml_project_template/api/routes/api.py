"""Defines the API routes"""

from fastapi import APIRouter
from ml_project_template.schemas import ModelRequest, ModelResponse
from ml_project_template.services import predict

router = APIRouter(prefix="/api", tags=["inference"])


# Include whatever routers you need here
@router.post("/predict", response_model=ModelResponse)
def run_inference(request: ModelRequest):
    """Performs inference using the loaded model"""
    return {"prediction": predict(request.request_input)}
