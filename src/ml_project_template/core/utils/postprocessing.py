from typing import Any, Protocol
from ml_project_template.core.configs.model_configs import CLASS_MAP  # type: ignore


class PostProcessor(Protocol):
    def __call__(self, data: Any) -> str: ...


########################
# Example Postprocessor #
########################
class CleanText:
    def __init__(self):
        pass

    def _clean_text(self, text: str) -> str:
        return text.strip()

    def __call__(self, data: str) -> str:
        return self._clean_text(data)


class Translate:
    def __init__(self) -> None:
        pass

    def __call__(self, data: int) -> str:
        return CLASS_MAP.get(data, "")


class PostProcessorPipeline:  # This should be called with a list of post-processors, and it will apply them sequentially to the input data
    def __init__(self, steps: list[PostProcessor]):
        self.steps = steps

    def __call__(self, data: int) -> str:
        for step in self.steps:
            data = step(data)  # type: ignore
        assert isinstance(data, str)
        return data
