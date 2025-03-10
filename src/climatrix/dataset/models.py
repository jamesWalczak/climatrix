from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from climatrix.dataset.base import BaseDataset


@dataclass(frozen=False)
class DatasetDefinition:
    name: str
    latitude_name: str
    longitude_name: str
    dataset: str
    time_name: str | None = None
    dataset_class: type[BaseDataset] = field(init=False)

    def __post_init__(self) -> None:
        module_name, class_name = self.dataset.rsplit(".", 1)
        module = importlib.import_module(module_name)
        self.dataset_class = getattr(module, class_name)
