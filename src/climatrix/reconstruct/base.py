from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from climatrix.dataset.dense import DenseDataset
    from climatrix.dataset.sparse import SparseDataset


class BaseReconstructor(ABC):
    __slots__ = ("dataset", "query_lat", "query_lon")
    _DEFAULT_LAT_RESOLUTION: ClassVar[float] = 0.1
    _DEFAULT_LON_RESOLUTION: ClassVar[float] = 0.1

    dataset: SparseDataset

    def __init__(
        self,
        dataset: SparseDataset,
        lat: slice = slice(-90, 90, 0.1),
        lon: slice = slice(-180, 180, 0.1),
    ) -> None:
        from climatrix.dataset.sparse import SparseDataset

        if not isinstance(dataset, SparseDataset):
            raise TypeError("Only SparseDataset object can be reconstructed")
        self.dataset = dataset
        if not isinstance(lat, slice):
            raise TypeError("lat must be a slice object")
        if not isinstance(lon, slice):
            raise TypeError("lon must be a slice object")
        if lat.step is None:
            lat = slice(lat.start, lat.stop, self._DEFAULT_LAT_RESOLUTION)
        if lon.step is None:
            lon = slice(lon.start, lon.stop, self._DEFAULT_LON_RESOLUTION)
        self.query_lat = lat
        self.query_lon = lon

    @abstractmethod
    def reconstruct(self) -> DenseDataset:
        raise NotImplementedError
