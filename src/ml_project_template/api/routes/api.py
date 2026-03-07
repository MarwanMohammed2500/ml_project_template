from fastapi import APIRouter
from src.ml_project_template.schemas import ModelRequest, ModelResponse
from .inference import predict

router = APIRouter(prefix="/api", tags=["inference"])


# Include whatever routers you need here
@router.post("/predict", response_model=ModelResponse)
def run_inference(request: ModelRequest):
    return {"prediction": predict(request.request_input)}
