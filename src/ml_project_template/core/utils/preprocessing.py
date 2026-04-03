from typing import Any, Protocol
import numpy as np
import numpy.typing as npt
import logging

logger = logging.getLogger(__name__)


class Preprocessor(Protocol):
    def __call__(self, data: Any) -> npt.NDArray[np.float32] | dict[str, Any]: ...


########################
# Example Preprocessor #
########################
class Normalizer:
    def __init__(self, mean: float | np.ndarray, std: float | np.ndarray):
        self.mean = mean
        self.std = std

    def __call__(
        self, data: npt.NDArray[np.float32] | dict[str, Any]
    ) -> npt.NDArray[np.float32] | dict[str, Any]:
        if isinstance(data, np.ndarray):
            return ((data - self.mean) / self.std).astype(np.float32)
        else:
            return {
                key: ((np.array(value) - self.mean) / self.std).astype(np.float32)
                for key, value in data.items()
            }


class PreprocessorPipeline:  # This should be called with a list of preprocessors, and it will apply them sequentially to the input data
    def __init__(self, steps: list[Preprocessor]):
        self.steps = steps

    def __call__(
        self, data: npt.NDArray[np.float32] | dict[str, Any]
    ) -> npt.NDArray[np.float32] | dict[str, Any]:
        assert len(self.steps) > 0, "Please Add some preprocessing steps first"
        processed_data = data
        for step in self.steps:
            processed_data = step(processed_data)
        return processed_data
