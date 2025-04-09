from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Self

import numpy as np
import xarray as xr
from matplotlib.axes import Axes

from climatrix.dataset.domain import Domain
from climatrix.decorators import cm_arithmetic_binary_operator

from .axis import Axis

# Based on MetPy
# (https://github.com/Unidata/MetPy/blob/main/src/metpy/xarray.py)
_coords_name_regex: dict[Axis, str] = {
    Axis.TIME: re.compile(r"^(x?)(valid_?)time(s?)([0-9]*)$"),
    Axis.VERTICAL: re.compile(
        r"^(z|lv_|bottom_top|sigma|h(ei)?ght|altitude|depth|"
        r"isobaric|pres|isotherm)"
        r"[a-z_]*[0-9]*$"
    ),
    Axis.LATITUDE: re.compile(r"^(x?)lat[a-z0-9_]*$"),
    Axis.LONGITUDE: re.compile(r"^(x?)lon[a-z0-9_]*$"),
    Axis.POINT: re.compile(r"^(point|points|values)$"),
}


@xr.register_dataset_accessor("cm")
class BaseClimatrixDataset(ABC):
    __slots__ = ("da", "_axis_mapping", "domain")

    da: xr.DataArray
    _axis_mapping: dict[Axis, str]

    def __new__(cls, xarray_obj: xr.Dataset | xr.DataArray) -> Self:
        cls._validate_input(xarray_obj)
        da = cls._ensure_single_var(xarray_obj)
        axis_mapping = cls._match_axis_names(da)
        cls._validate_spatial_axes(axis_mapping)

        if cls is BaseClimatrixDataset:
            if (
                Axis.TIME in axis_mapping
                and da[axis_mapping[Axis.TIME]].size > 1
            ):
                if cls._check_is_dense(da, axis_mapping):
                    from .dense import DynamicDenseDataset

                    return DynamicDenseDataset(da)
                else:
                    from .sparse import DynamicSparseDataset

                    return DynamicSparseDataset(da)
            else:
                if cls._check_is_dense(da, axis_mapping):
                    from .dense import StaticDenseDataset

                    return StaticDenseDataset(da)
                else:
                    from .sparse import StaticSparseDataset

                    return StaticSparseDataset(da)
        return super().__new__(cls)

    @staticmethod
    def _validate_input(da: xr.Dataset | xr.DataArray):
        if not isinstance(da, (xr.Dataset, xr.DataArray)):
            raise NotImplementedError(
                "At the moment, dataset can be created only based on "
                "xarray.DataArray or single-variable xarray.Dataset "
                f"objects, but provided {type(da).__name__}"
            )

    @staticmethod
    def _ensure_single_var(da: xr.Dataset | xr.DataArray) -> xr.DataArray:
        if isinstance(da, xr.Dataset):
            if len(da.data_vars) > 1:
                raise ValueError(
                    "Dataset can be created only based on "
                    "xarray.DataArray or xarray.Dataset with single variable "
                    "objects, but provided xarray.Dataset with multiple "
                    "data_vars."
                )
            return da[list(da.data_vars.keys())[0]]
        return da

    @staticmethod
    def _match_axis_names(da: xr.DataArray) -> dict[Axis, str]:
        # TODO: should be moved to Domain class
        axis_names = {}
        coords_and_dims = {*da.dims, *da.coords.keys()}
        for coord in coords_and_dims:
            for axis, regex in _coords_name_regex.items():
                if regex.match(coord):
                    axis_names[axis] = coord
                    break
        return axis_names

    @staticmethod
    def _validate_spatial_axes(axis_mapping: dict[Axis, str]):
        # TODO: should be moved to Domain class
        for axis in [Axis.LATITUDE, Axis.LONGITUDE]:
            if axis not in axis_mapping:
                raise ValueError(f"Dataset has no {axis.name} axis")

    @staticmethod
    def _check_is_dense(
        da: xr.DataArray, axis_mapping: dict[Axis, str]
    ) -> bool:
        return (axis_mapping[Axis.LATITUDE] in da.dims) and (
            axis_mapping[Axis.LONGITUDE] in da.dims
        )

    def __init__(self, xarray_obj: xr.DataArray):
        self.da = self._ensure_single_var(xarray_obj)
        self.axis_mapping = self._match_axis_names(self.da)
        self._update_domain()

    def _update_domain(self) -> None:
        self.domain = Domain(
            {
                axis: self.da[axis_name].values
                for axis, axis_name in self.axis_mapping.items()
            }
        )

    # ###############################
    #  Properties
    # ###############################

    @property
    def latitude_name(self) -> str:
        return self.axis_mapping[Axis.LATITUDE]

    @property
    def longitude_name(self) -> str:
        return self.axis_mapping[Axis.LONGITUDE]

    @property
    def time_name(self) -> str | None:
        if Axis.TIME not in self.axis_mapping:
            return None
        return self.axis_mapping[Axis.TIME]

    @property
    @lru_cache(maxsize=1)
    def latitude(self) -> xr.DataArray:
        return self.da[self.latitude_name]

    @property
    @lru_cache(maxsize=1)
    def longitude(self) -> xr.DataArray:
        return self.da[self.longitude_name]

    @property
    @lru_cache(maxsize=1)
    def time(self) -> xr.DataArray:
        if self.time_name is None:
            raise AttributeError("The dataset has no time dimension")
        return self.da[self.time_name]

    @property
    @lru_cache(maxsize=1)
    def is_dynamic(self) -> bool:
        return self.time_name and self.time.size > 1

    @property
    @lru_cache(maxsize=1)
    def size(self) -> int:
        time_size = self.time.size if self.time_name else 1
        return self.latitude.size * self.longitude.size * time_size

    # ###############################
    #  Operators
    # ###############################
    @cm_arithmetic_binary_operator
    def __add__(self, other: Any) -> Self: ...  # noqa: E704

    @cm_arithmetic_binary_operator
    def __sub__(self, other: Any) -> Self: ...  # noqa: E704

    @cm_arithmetic_binary_operator
    def __mul__(self, other: Any) -> Self: ...  # noqa: E704

    @cm_arithmetic_binary_operator
    def __truediv__(self, other: Any) -> Self: ...  # noqa: E704

    # ###############################
    #  Abstract methods
    # ###############################

    @abstractmethod
    def plot(
        self,
        title: str | None = None,
        target: str | os.PathLike | Path | None = None,
        show: bool = True,
        **kwargs,
    ) -> Axes:
        raise NotImplementedError

    def sel_time(
        self, time: datetime | np.datetime64 | slice | list | np.ndarray
    ) -> Self:
        return type(self)(
            self.da.sel({self.time_name: time}, method="nearest")
        )

    def isel_time(self, time: int | list[int] | np.ndarray | slice) -> Self:
        if self.time_name and self.time.size > 1:
            return type(self)(self.da.isel({self.time_name: time}))
        return self
