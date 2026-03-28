from src.ml_project_template.model import SupervisedModel
from src.ml_project_template.configs.model_configs import (
    MODEL_PATH,
    MODEL_TYPE,
    TASK_TYPE,
    CLASS_MAP,
    DECISION_THRESHOLD,
)

model = None


def load_model():
    """Checks if the model is loaded, if not, loads it using the model configurations"""
    global model
    if model is None:
        model = SupervisedModel(
            model_path=MODEL_PATH,
            model_type=MODEL_TYPE,
            task_type=TASK_TYPE,
            class_map=CLASS_MAP,
            decision_threshold=DECISION_THRESHOLD,
        )
        model.preload()


def predict(request_input):
    """performs prediction using the loaded model"""
    return model.predict(input=request_input)
