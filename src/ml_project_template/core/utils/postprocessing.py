from typing import Any, Protocol


class PostProcessor(Protocol):
    def __call__(self, data: Any) -> str: ...


########################
# Example Postprocessor #
########################
class CleanText:
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = mean
        self.std = std

    def _clean_text(self, text: str) -> str:
        return text.strip()

    def __call__(self, data: str) -> str:
        return self._clean_text(data)


class PostProcessorPipeline:  # This should be called with a list of post-processors, and it will apply them sequentially to the input data
    def __init__(self, steps: list[PostProcessor]):
        self.steps = steps

    def __call__(self, data: str) -> str:
        for step in self.steps:
            data = step(data)
        return data
