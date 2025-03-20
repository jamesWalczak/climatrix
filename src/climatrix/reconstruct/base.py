from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

import numpy as np

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
        lat: slice | np.ndarray = slice(-90, 90, _DEFAULT_LAT_RESOLUTION),
        lon: slice | np.ndarray = slice(-180, 180, _DEFAULT_LON_RESOLUTION),
    ) -> None:
        self._validate_types(dataset, lat, lon)
        self.dataset = dataset
        self.query_lat = self._maybe_get_lat_from_slice(lat)
        self.query_lon = self._maybe_get_lon_from_slice(lon)

    def _validate_types(self, dataset, lat, lon) -> None:
        from climatrix.dataset.sparse import SparseDataset

        if not isinstance(dataset, SparseDataset):
            raise TypeError("dataset must be a SparseDataset object")
        if not isinstance(lat, (slice, np.ndarray)):
            raise TypeError("lat must be a slice object or a NumPy array")
        if not isinstance(lon, (slice, np.ndarray)):
            raise TypeError("lon must be a slice object or a NumPy array")

    def _maybe_get_lat_from_slice(self, lat: slice | np.ndarray) -> np.ndarray:
        if isinstance(lat, slice):
            lat = np.arange(
                lat.start,
                lat.stop,
                (
                    lat.step
                    if lat.step is not None
                    else self._DEFAULT_LAT_RESOLUTION
                ),
            )
        return lat

    def _maybe_get_lon_from_slice(self, lon: slice | np.ndarray) -> np.ndarray:
        if isinstance(lon, slice):
            lon = np.arange(
                lon.start,
                lon.stop,
                (
                    lon.step
                    if lon.step is not None
                    else self._DEFAULT_LON_RESOLUTION
                ),
            )
        return lon

    @abstractmethod
    def reconstruct(self) -> DenseDataset:
        raise NotImplementedError
