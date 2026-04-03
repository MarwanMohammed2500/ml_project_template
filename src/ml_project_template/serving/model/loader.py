from typing import Any, Literal, Optional
import numpy as np
import numpy.typing as npt
import onnxruntime as ort  # type: ignore
from scipy.special import softmax  # type: ignore
from ml_project_template.core.errors import InvalidModelPathError  # type: ignore
from ml_project_template.core.utils import PreprocessorPipeline  # type: ignore
from mlflow.artifacts import download_artifacts
import re


class Model:
    """Base Class for all model classes"""

    def __init__(
        self,
        model_uri: str,
        task_type: Literal["binary", "multiclass", "regression"] | None = None,
        preproc_pipeline: Optional[PreprocessorPipeline] = None,
        *args: Any,
        **kwargs: Any,
    ):
        self.model_uri = model_uri
        self.task_type = task_type
        self._model = None
        self.args = args
        self.kwargs = kwargs
        self._strategy = None

        self.preproc_pipeline = preproc_pipeline

        if task_type not in ["binary", "multiclass", "regression"]:
            raise ValueError(
                "Invalid task type, supported values are: 'binary', 'multiclass', 'regression'"
            )

    @property
    def loaded(self):
        """
        A flag to determine if the model is loaded or not
        """
        return self._model is not None

    @property
    def raw(self):
        """
        Use the raw model to access model specific methods and attributes
        """
        self.preload()
        return self._model

    def _verify_model_uri(self) -> bool:
        if not re.compile("^models:/[a-zA-Z0-9_-]+@production$").match(self.model_uri):
            return False
        return True

    def _load_model(self):
        if self._verify_model_uri():
            local_path = download_artifacts(artifact_uri=self.model_uri)
            self._model = ort.InferenceSession(
                f"{local_path}/model.onnx",
                providers=[
                    "CPUExecutionProvider"
                ],  # if working with Nvidia GPU, install onnxruntime-gpu and add "CUDAExecutionProvider" as the first item in this list
            )
        else:
            raise InvalidModelPathError(
                "The model path is invalid, please verify if the model has the correct extention, or that the path exists."
            )

    def preload(self):
        """
        loads the model if it is not loaded yet. If not called, the model is first loaded at the first inference call
        """
        if self._model is None:
            self._load_model()

        if self._strategy is None:
            assert self._model is not None
            if self.task_type == "binary":
                self._strategy = _BinaryClassifierModel(
                    model_instance=self._model,
                    decision_threshold=self.kwargs.get("decision_threshold", 0.5),
                )
            elif self.task_type == "multiclass":
                self._strategy = _MulticlassClassifierModel(model_instance=self._model)
            elif self.task_type == "regression":
                raise NotImplementedError("Regression task is not implemented yet")
            else:
                raise ValueError(
                    "Invalid task type, supported values are: 'binary', 'multiclass', 'regression'"
                )

    def predict(self, **kwargs: Any) -> tuple[int, float]:
        self.preload()
        assert self._strategy is not None
        if self.preproc_pipeline is not None:
            preproc_inputs = {
                name: kwargs[name]
                for name in self._strategy.input_names
                if name in kwargs
            }
            preprocessed_data = self.preproc_pipeline(preproc_inputs)
            kwargs.update(preprocessed_data)
        return self._strategy.predict(**kwargs)


class _BinaryClassifierModel:
    """
    This class acts as an abstraction layer for a binary classifier model loading, inference,
    and logits processing

    Args:
        mode_instance: ort.InferenceSession:
            The loaded model instance
    """

    def __init__(
        self,
        model_instance: ort.InferenceSession,
        decision_threshold: float = 0.5,
    ):
        self.decision_threshold = decision_threshold
        self.model_instance = model_instance
        self.input_names: list[str] = [inp.name for inp in model_instance.get_inputs()]  # type: ignore
        self.output_name: str = model_instance.get_outputs()[0].name  # type: ignore

    def sigmoid(self, x: npt.NDArray[np.float32]) -> float:
        return float((1 / (1 + np.exp(-x))).item())

    def _compare_logits_and_threshold(
        self, logits: npt.NDArray[np.float32]
    ) -> tuple[int, float]:
        prob = self.sigmoid(logits)
        pred = int(prob > self.decision_threshold)
        return pred, prob

    def _process_model_output(
        self, logits: npt.NDArray[np.float32]
    ) -> tuple[int, float]:
        pred, prob = self._compare_logits_and_threshold(logits)
        return pred, prob

    def predict(self, **kwargs: Any) -> tuple[int, float]:
        """perform inference using the loaded model"""
        output, prob = None, None
        assert hasattr(self.model_instance, "run")
        ort_inputs = {name: kwargs[name] for name in self.input_names if name in kwargs}
        logits: np.ndarray = self.model_instance.run(  # type: ignore
            [self.output_name], ort_inputs
        )[0]
        print(f"logits = {logits}")
        output, prob = self._process_model_output(np.array(logits, dtype=np.float32))
        return output, prob


class _MulticlassClassifierModel:
    """
    This class acts as an abstraction layer for a multiclass classifier model loading, inference,
    and logits processing

    Args:
        mode_instance: ort.InferenceSession:
            The loaded model instance
    """

    def __init__(
        self,
        model_instance: ort.InferenceSession,
    ):
        self.model_instance = model_instance
        self.input_names: list[str] = [inp.name for inp in model_instance.get_inputs()]  # type: ignore
        self.output_name: str = model_instance.get_outputs()[0].name  # type: ignore

    def _process_model_output(
        self, logits: npt.NDArray[np.float32]
    ) -> tuple[int, float]:
        probs = softmax(logits, axis=-1)
        pred = int(np.argmax(probs, axis=-1).item())

        prob = float(probs.squeeze()[pred])
        return pred, prob

    def predict(self, **kwargs: Any) -> tuple[int, float]:
        """perform inference using the loaded model"""
        output, prob = None, None
        assert hasattr(self.model_instance, "run")
        ort_inputs = {name: kwargs[name] for name in self.input_names if name in kwargs}
        logits: np.ndarray = self.model_instance.run(  # type: ignore
            [self.output_name], ort_inputs
        )[0]
        output, prob = self._process_model_output(np.array(logits, dtype=np.float32))
        return output, prob
