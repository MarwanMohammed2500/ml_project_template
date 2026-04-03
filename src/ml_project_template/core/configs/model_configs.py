# Example Configurations
from ml_project_template.core.utils.preprocessing import (
    Normalizer,
    PreprocessorPipeline,
)  # type: ignore

MODEL_TYPE: str = "onnx"  # or pt
TASK_TYPE: str = "binary"  # or /multiclass/regression
CLASS_MAP: dict[int, str] = {0: "your", 1: "class", 2: "map"}
DECISION_THRESHOLD: float = 0.5
PREPROC_PIPELINE: PreprocessorPipeline = PreprocessorPipeline(
    steps=[Normalizer(mean=0.5, std=0.5)]
)  # Example pipeline, replace with your actual preprocessors

PRODUCTION_MODEL_URI: str = "models:/SimpleModel_ONNX@production"
