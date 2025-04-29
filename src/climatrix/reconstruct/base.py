from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from climatrix.dataset.domain import Domain

if TYPE_CHECKING:
    from climatrix.dataset.base import BaseClimatrixDataset


class BaseReconstructor(ABC):
    __slots__ = ("dataset", "query_lat", "query_lon")
    _DEFAULT_LAT_RESOLUTION: ClassVar[float] = 0.1
    _DEFAULT_LON_RESOLUTION: ClassVar[float] = 0.1

    dataset: BaseClimatrixDataset

    def __init__(
        self, dataset: BaseClimatrixDataset, target_domain: Domain
    ) -> None:
        self.dataset = dataset
        self.target_domain = target_domain

    def _validate_types(self, dataset, lat, lon) -> None:
        from climatrix.dataset.base import BaseClimatrixDataset

        if not isinstance(dataset, BaseClimatrixDataset):
            raise TypeError("dataset must be a BaseClimatrixDataset object")

    @abstractmethod
    def reconstruct(self) -> BaseClimatrixDataset:
        raise NotImplementedError
