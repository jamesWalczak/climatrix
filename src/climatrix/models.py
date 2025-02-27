from typing import Any
from pydantic import BaseModel

class Request(BaseModel):
    dataset: str
    request: dict[str, Any]