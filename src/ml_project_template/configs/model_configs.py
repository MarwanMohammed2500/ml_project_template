# Example Configurations
from ml_project_template.utils.preprocessing import Normalizer, PreprocessorPipeline  # type: ignore

MODEL_TYPE = "onnx"  # or pt
TASK_TYPE = "binary"  # or /multiclass/regression
CLASS_MAP = {0: "your", 1: "class", 2: "map"}
DECISION_THRESHOLD = 0.5
PREPROC_PIPELINE = PreprocessorPipeline(
    steps=[Normalizer(mean=0.5, std=0.5)]
)  # Example pipeline, replace with your actual preprocessors
PRODUCTION_MODEL_URI = 'models:/SimpleModel_ONNX@production'