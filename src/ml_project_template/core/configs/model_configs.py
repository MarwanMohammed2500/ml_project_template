# Example Configurations
from typing import Literal

MODEL_TYPE: str = "onnx"  # or pt
TASK_TYPE: Literal["binary", "multiclass", "regression"] = "binary"
CLASS_MAP: dict[int, str] = {0: "your", 1: "class", 2: "map"}
DECISION_THRESHOLD: float = 0.5

PRODUCTION_MODEL_URI: str = "models:/SimpleModel_ONNX@production"
