from pydantic import BaseModel
from typing import Any


# Customize this class per model
class ModelResponse(BaseModel):
    prediction: Any
