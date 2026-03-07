from src.ml_project_template.errors.exceptions import InvalidModelPathError
from typing import Literal, Optional, Any
from scipy.special import softmax
import numpy.typing as npt
import onnxruntime as ort
import numpy as np
import torch
import os


class SupervisedModel:
    """
    This class acts as an abstraction layer for supervised model loading, inference, and logits processing

    Args:
        model_path: str:
            The path to the model's file

        model_type: Literal["pt", "onnx"]:
            Whether you're loading a torch model or an onnx model

        task_type: Literal["binary_classification", "multiclass_classification", "regression"]:
            The task that the model performs

        class_map: dict[int, str]:
            The class map/label map of the classifier

        decision_threshold: Optional[float] = None:
            The threshold to make a decision in binary classifiers, leave as None if task_type != binary_classification. If you don't set it for binary classifiers, it defaults to 0.5

        torch_weights_only: bool = False:
            Whether to load only weights of the model or to load the full model graph (never set this to true if you don't know the source of the model)
    """

    def __init__(
        self,
        model_path: str,
        model_type: Literal["pt", "onnx"],
        task_type: Literal[
            "binary_classification", "multiclass_classification", "regression"
        ],
        class_map: dict[int, str],
        decision_threshold: Optional[float] = None,
        torch_weights_only: bool = False,
    ):
        self.torch_weights_only = torch_weights_only
        self.decision_threshold = decision_threshold
        self.model_path = model_path
        self.model_type = model_type
        self.task_type = task_type
        self.class_map = class_map

        self._model = None

        if self.model_type == "pt":
            self._target_ext = ".pt"
        elif self.model_type == "onnx":
            self._target_ext = ".onnx"
        else:
            self._target_ext = None

        if self.task_type == "binary_classification" and self.decision_threshold is None:
            self.decision_threshold = 0.5

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
        self._preload()
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
                self._input_name = self._model.get_inputs()[0].name
                self._output_name = self._model.get_outputs()[0].name
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

    def _process_classifier_output(self, logits: npt.NDArray[float]) -> (str, float):
        logits = np.array(
            logits
        )  # So if the logits are torch tensors or numpy arrays it gets treated as a numpy array
        if self.task_type == "binary_classification":
            prob = float(1 / (1 + np.exp(-logits)))  # Sigmoid function
            pred = int(prob > self.decision_threshold)
            output_class = self.class_map[pred]
        elif self.task_type == "multiclass_classification":
            probs = softmax(logits, axis=-1)
            pred = int(np.argmax(probs, axis=-1).item())
            prob = float(probs.squeeze()[pred])
            output_class = self.class_map[pred]
        return output_class, prob

    def predict(self, *args: Any, **kwargs: Any):
        self.preload()
        if self.model_type == "onnx":
            return self._process_classifier_output(
                self._model.run(
                    [self._output_name], {self._input_name: kwargs["input"]}
                )[0]
            )
        elif self.model_type == "pt":
            self._model.eval()
            with torch.inference_mode():
                return self._process_classifier_output(self._model(*args, **kwargs))
