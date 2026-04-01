import os
from typing import Any, Literal
import numpy as np
import numpy.typing as npt
import onnxruntime as ort # type: ignore
import torch
from scipy.special import softmax # type: ignore
from src.ml_project_template.errors import InvalidModelPathError


class _Model:
    """Base Class for all model classes"""

    def __init__(
        self,
        model_path: str,
        model_type: Literal["pt", "onnx"],
        torch_weights_only: bool = False,
    ):
        self.torch_weights_only = torch_weights_only
        self.model_path = model_path
        self.model_type = model_type

        self._model = None
        self._input_name = None
        self._output_name = None

        if self.model_type == "pt":
            self._target_ext = ".pt"
        elif self.model_type == "onnx":
            self._target_ext = ".onnx"
        else:
            self._target_ext = None

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

    def _verify_model_path(self) -> bool:
        if not os.path.exists(self.model_path):
            return False
        if self._target_ext is not None:
            if self.model_path.endswith(self._target_ext):
                return True
        return False

    def _load_model(self):
        if self._verify_model_path():
            if self.model_type == "onnx":
                self._model = ort.InferenceSession(
                    self.model_path,
                    providers=[
                        "CPUExecutionProvider"
                    ],  # if working with Nvidia GPU, install onnxruntime-gpu and add "CUDAExecutionProvider" as the first item in this list
                )
                self._input_name = self._model.get_inputs()[0].name # type: ignore
                self._output_name = self._model.get_outputs()[0].name # type: ignore
            elif self.model_type == "pt":
                self._model = torch.load(
                    self.model_path, weights_only=self.torch_weights_only
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


class SupervisedModel(_Model):
    """
    This class acts as an abstraction layer for supervised model loading, inference,
    and logits processing

    Args:
        model_path: str:
            The path to the model's file

        model_type: Literal["pt", "onnx"]:
            Whether you're loading a torch model or an onnx model

        task_type: Literal["binary", "multiclass", "regression"]:
            The task that the model performs

        class_map: dict[int, str]:
            The class map/label map of the classifier

        decision_threshold: Optional[float] = None:
            The threshold to make a decision in binary classifiers.
            Leave as None if task_type != binary.
            If you don't set it for binary classifiers, it defaults to 0.5

        torch_weights_only: bool = False:
            Whether to load only weights of the model or to load the full model graph
            **(never set this to true if you don't know the source of the model)**
    """

    def __init__(
        self,
        task_type: Literal["binary", "multiclass", "regression"],
        class_map: dict[int, str],
        model_path: str,
        model_type: Literal["pt", "onnx"],
        torch_weights_only: bool = False,
        decision_threshold: float = 0.5,
    ):
        super().__init__(
            model_path=model_path,
            model_type=model_type,
            torch_weights_only=torch_weights_only
        )
        self.decision_threshold = decision_threshold
        self.task_type = task_type
        self.class_map = class_map
    
    def sigmoid(self, x: npt.NDArray[np.float32]) -> float:
        return float(1 / (1 + np.exp(-x)))
    
    def _compare_logits_and_threshold(self, logits: npt.NDArray[np.float32]) -> tuple[int, float]:
        prob = self.sigmoid(logits)
        pred = int(prob > self.decision_threshold)
        return pred, prob
    
    def _process_classifier_output(self, logits: npt.NDArray[np.float32]) -> tuple[str, float] | tuple[None, None]:
        if self.task_type == "binary":
            pred, prob = self._compare_logits_and_threshold(logits)
        elif self.task_type == "multiclass":
            probs = softmax(logits, axis=-1)
            pred = int(np.argmax(probs, axis=-1).item())
            prob = float(probs.squeeze()[pred])
        else:
            return None, None
        output_class = self.class_map[pred]
        return output_class, prob

    def predict(self, *args: Any, **kwargs: Any) -> tuple[Any | None, float | None]:
        """perform inference using the loaded model"""
        self.preload()
        output, prob = None, None
        if self.model_type == "onnx":
            output, prob = self._process_classifier_output(
                self._model.run([self._output_name], {self._input_name: kwargs["input"]})[0] # type: ignore
            )
        elif self.model_type == "pt":
            self._model.eval() # type: ignore
            with torch.inference_mode():
                output, prob = self._process_classifier_output(self._model(*args, **kwargs).detach().cpu().numpy()) # type: ignore
        return output, prob
