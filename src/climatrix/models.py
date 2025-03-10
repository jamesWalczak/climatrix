from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Request:
    dataset: str
    request: dict[str, Any]
    filename: Path
