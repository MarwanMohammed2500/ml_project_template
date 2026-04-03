from pydantic import BaseModel
from typing import Any


# Customize this class per model
class ModelRequest(BaseModel):
    request_id: str
    request_input: Any
