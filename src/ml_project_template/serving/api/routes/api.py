"""Defines the API routes"""

from fastapi import APIRouter
from ml_project_template.core.schemas import ModelRequest, ModelResponse  # type: ignore
from ml_project_template.serving.services import predict  # type: ignore
from typing import Any

router = APIRouter(prefix="/api", tags=["inference"])


# Include whatever routers you need here
@router.post("/predict", response_model=ModelResponse)
def run_inference(request: ModelRequest) -> dict[str, Any]:
    """Performs inference using the loaded model"""
    return {"code": 200, "prediction": predict(request.request_input)}
