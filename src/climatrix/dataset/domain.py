from typing import Any

import numpy as np

from climatrix.dataset.axis import Axis


class Domain:

    __slots__ = ("coords",)
    coords: dict[str, np.ndarray]

    def __init__(self, coords: dict[Axis, np.ndarray]):
        self.coords = coords
        self._ensure_all_numpy_arrays()
        self._filter_out_single_value_coord()

    def _ensure_all_numpy_arrays(self) -> None:
        self.coords = {k: np.array(v, ndmin=1) for k, v in self.coords.items()}

    def _filter_out_single_value_coord(self):
        self.coords = {k: v for k, v in self.coords.items() if len(v) > 1}

    @property
    def latitude(self) -> np.ndarray:
        if Axis.LATITUDE not in self.coords:
            raise ValueError(
                f"Latitude not found in coordinates {self.coords.keys()}"
            )
        return self.coords[Axis.LATITUDE]

    @property
    def longitude(self) -> np.ndarray:
        if Axis.LONGITUDE not in self.coords:
            raise ValueError(
                f"Longitude not found in coordinates {self.coords.keys()}"
            )
        return self.coords[Axis.LONGITUDE]

    @property
    def size(self) -> int:
        if Axis.POINT in self.coords:
            return self.coords[Axis.POINT].size
        return self.latitude.size * self.longitude.size

    def __eq__(self, value: Any) -> bool:
        if not isinstance(value, Domain):
            return False
        same_keys = set(self.coords.keys()) == set(value.coords.keys())
        if not same_keys:
            return False
        for k in self.coords.keys():
            if not np.array_equal(self.coords[k], value.coords[k]):
                return False
        return True
