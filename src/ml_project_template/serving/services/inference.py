from ml_project_template.serving.model import Model  # type: ignore
from typing import Any
from ml_project_template.core.configs.model_configs import (  # type: ignore
    MODEL_TYPE,
    TASK_TYPE,
    CLASS_MAP,
    DECISION_THRESHOLD,
    PRODUCTION_MODEL_URI,
)
from ml_project_template.core.utils import (  # type: ignore
    PreprocessorPipeline,
    Normalizer,
    PostProcessorPipeline,
    CleanText,
    Translate,
)  # type: ignore
import numpy as np
from mlflow.artifacts import download_artifacts
import logging
import pickle
import json
import os

logger = logging.getLogger(__name__)

model = None


def load_assets() -> PreprocessorPipeline:
    """Load the model assets (Pickle objects, JSON fallbacks)

    Args:
        None

    Returns:
        pipeline: PreprocessorPipeline:
            The preprocessing pipeline object
    """
    local_dir = download_artifacts(artifact_uri=PRODUCTION_MODEL_URI)
    pipeline_path = None
    for root, _, files in os.walk(local_dir):
        for f in files:
            if f.endswith(".pkl"):
                pipeline_path = os.path.join(root, f)
    try:
        with open(pipeline_path, "rb") as f:  # type: ignore
            pipeline = pickle.load(f)  # type: ignore
    except (pickle.UnpicklingError, ImportError, IndexError, TypeError) as e:
        # PLAN B: Reconstruct from JSON constants
        logger.warning(
            f"Pickle load failed. Attempting reconstruction from constants...\nerror: {e}",
            extra={"error_details": e},
        )
        constants_path = os.path.join(local_dir, "normalization_constants.json")
        with open(constants_path, "r") as f:
            constants = json.load(f)

        # Manually rebuild the pipeline
        pipeline = PreprocessorPipeline(
            steps=[
                Normalizer(
                    mean=np.array(constants["norm_mean"]),
                    std=np.array(constants["norm_std"]),
                )
            ]
        )

    return pipeline


def load_model():
    """Checks if the model is loaded, if not, loads it using the model configurations"""
    global model
    if model is None:
        assert TASK_TYPE in ["binary", "multiclass", "regression"]
        preproc_pipeline = load_assets()
        model = Model(
            model_uri=PRODUCTION_MODEL_URI,
            model_type=MODEL_TYPE,
            task_type=TASK_TYPE,
            class_map=CLASS_MAP,
            decision_threshold=DECISION_THRESHOLD,
            preproc_pipeline=preproc_pipeline,
        )
        model.preload()


post_processor = PostProcessorPipeline(steps=[Translate(), CleanText()])


def predict(request_input: Any) -> tuple[Any, float]:
    """performs prediction using the loaded model

    Args:
        request_input: Any:
            The input from the request

    Returns:
        output: Any:
            The model's output

        prob: float:
            Model confidence
    """
    assert model is not None, (
        "Model is not loaded, please call load_model() before inference"
    )
    output, prob = model.predict(x=request_input)
    output = post_processor(output)
    return output, prob
