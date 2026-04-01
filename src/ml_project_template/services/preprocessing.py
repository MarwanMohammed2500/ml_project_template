from typing import Any, Protocol
import numpy as np
import numpy.typing as npt

class Preprocessor(Protocol):
    def __call__(self, data: Any) -> dict[str, npt.NDArray[np.float32]]:
        ...

########################
# Example Preprocessor #
########################
class Normalizer:
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = mean
        self.std = std

    def __call__(self, data: dict[str, npt.NDArray[np.float32]]) -> dict[str, npt.NDArray[np.float32]]:
        return {key: (value - self.mean) / self.std for key, value in data.items()}

class Pipeline: # This should be called with a list of preprocessors, and it will apply them sequentially to the input data
    def __init__(self, steps: list[Preprocessor]):
        self.steps = steps

    def __call__(self, data: dict[str, npt.NDArray[np.float32]]) -> dict[str, npt.NDArray[np.float32]]:
        for step in self.steps:
            data = step(data)
        return data