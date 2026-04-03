from ml_project_template.serving.model import Model  # type: ignore
from typing import Any
from ml_project_template.core.configs.model_configs import (  # type: ignore
    MODEL_TYPE,
    TASK_TYPE,
    CLASS_MAP,
    DECISION_THRESHOLD,
    PREPROC_PIPELINE,
    PRODUCTION_MODEL_URI,
)
from ml_project_template.core.utils import PostProcessorPipeline, CleanText  # type: ignore


model = None


def load_model():
    """Checks if the model is loaded, if not, loads it using the model configurations"""
    global model
    if model is None:
        model = Model(
            model_uri=PRODUCTION_MODEL_URI,
            model_type=MODEL_TYPE,
            task_type=TASK_TYPE,
            class_map=CLASS_MAP,
            decision_threshold=DECISION_THRESHOLD,
            preproc_pipeline=PREPROC_PIPELINE,
        )
        model.preload()


post_processor = PostProcessorPipeline(steps=[CleanText()])


def predict(request_input: Any) -> tuple[Any, float]:
    """performs prediction using the loaded model"""
    assert model is not None, (
        "Model is not loaded, please call load_model() before inference"
    )
    output, prob = model.predict(input=request_input)
    output = CLASS_MAP[output]
    output = post_processor(output)  # type:ignore - here, output is NOT a string, you can do whatever you want based on the project's needs
    return output, prob
