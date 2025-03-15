from __future__ import annotations

from functools import lru_cache
import re
from pathlib import Path
import os
from abc import ABC,  abstractmethod
from typing import Self

import xarray as xr
from .axis import Axis
from climatrix.dataset import axis

# Based on MetPy (https://github.com/Unidata/MetPy/blob/main/src/metpy/xarray.py)
_coords_name_regex: dict[Axis, str] ={
        Axis.TIME: re.compile(r'^(x?)(valid_?)time(s?)([0-9]*)$'),
        Axis.VERTICAL: re.compile(
            r'^(z|lv_|bottom_top|sigma|h(ei)?ght|altitude|depth|isobaric|pres|isotherm)'
            r'[a-z_]*[0-9]*$'
        ),
        Axis.LATITUDE: re.compile(r'^(x?)lat[a-z0-9_]*$'),
        Axis.LONGITUDE: re.compile(r'^(x?)lon[a-z0-9_]*$'),
        Axis.POINT: re.compile(r"^scatter_(data|points|values)_([a-zA-Z0-9_]+)(_v[0-9]+)?\.csv$") 
}


@xr.register_dataset_accessor("cm")
class BaseClimatrixDataset(ABC):
    __slots__ = ("da", "_axis_mapping")

    da: xr.DataArray
    _axis_mapping: dict[Axis, str]

    def __new__(cls, xarray_obj: xr.Dataset | xr.DataArray) -> Self:
        cls._validate_input(xarray_obj)
        da = cls._ensure_single_var(xarray_obj)
        axis_mapping = cls._match_axis_names(da)
        cls._validate_spatial_axes(axis_mapping)

        if cls is BaseClimatrixDataset:
            if Axis.TIME in axis_mapping and da[axis_mapping[Axis.TIME]].size > 1:
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
        axis_names = {}
        for coord in da.coords:
            for axis, regex in _coords_name_regex.items():
                if regex.match(coord):
                    axis_names[axis] = coord
                    break
        return axis_names
    
    @staticmethod
    def _validate_spatial_axes(axis_mapping: dict[Axis, str]):
        for axis in [Axis.LATITUDE, Axis.LONGITUDE]:
            if axis not in axis_mapping:
                raise ValueError(
                    f"Dataset has no {axis.name} axis"
                )
    
    @staticmethod
    def _check_is_dense(da: xr.DataArray, axis_mapping: dict[Axis, str]) -> bool:
        return (axis_mapping[Axis.LATITUDE] in da.dims) and (axis_mapping[Axis.LONGITUDE] in da.dims)
    

    def __init__(self, xarray_obj: xr.DataArray):
        self.da = xarray_obj
        self.axis_mapping = self._match_axis_names(self.da)

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
            raise AttributeError(
                f"The dataset has no time dimension"
            )
        return self.da[self.time_name]
    
    @abstractmethod
    def plot(self, target: str | os.PathLike | Path | None = None, show: bool = False, **kwargs) -> None:
        raise NotImplementedError




















    # da: xr.DataArray
    # _def: DatasetDefinition

    # def __init__(
    #     self, da: xr.Dataset | xr.DataArray, definition: DatasetDefinition
    # ) -> None:
    #     if not isinstance(definition, DatasetDefinition):
    #         raise TypeError(
    #             "Definition can be created only based on "
    #             "DatasetDefinition object, "
    #             f"but provided {type(definition).__name__}"
    #         )
    #     if isinstance(da, xr.Dataset):
    #         if len(da.data_vars) > 1:
    #             raise ValueError(
    #                 "Dataset can be created only based on "
    #                 "xarray.DataArray or xarray.Dataset with single variable "
    #                 "objects, but provided xarray.Dataset with multiple "
    #                 "data_vars."
    #             )
    #         da = da[list(da.data_vars.keys())[0]]
    #     if not isinstance(da, xr.DataArray):
    #         raise TypeError(
    #             "Dataset can be created only based on "
    #             "xarray.DataArray or single-variable xarray.Dataset objects, "
    #             f"but provided {type(da).__name__}"
    #         )
    #     self.da = da
    #     self._def = definition
    #     self.validate()

    # @property
    # def latitude(self) -> xr.DataArray:
    #     return self.da[self._def.latitude_name]

    # @property
    # def longitude(self) -> xr.DataArray:
    #     return self.da[self._def.longitude_name]

    # @property
    # def time(self) -> xr.DataArray:
    #     if self._def.time_name is None:
    #         raise AttributeError(
    #             f"The dataset {self._def.name} has no time dimension"
    #         )
    #     return self.da[self._def.time_name]

    # @abstractmethod
    # def validate(self) -> None:
    #     raise NotImplementedError

    # @abstractmethod
    # def plot(self, ax: Axes | None = None, **kwargs) -> Axes:
    #     raise NotImplementedError

    # @classmethod
    # def load(
    #     cls,
    #     path: str | os.PathLike | Path,
    #     definition_file: str | os.PathLike | Path | None = None,
    # ) -> Self:
    #     path = Path(path).expanduser().resolve()
    #     if not path.exists():
    #         raise FileNotFoundError(f"The file {path} does not exist")
    #     if definition_file is None:
    #         definition_file: Path = path.with_suffix(path.suffix + ".def")
    #         if not definition_file.exists():
    #             raise FileNotFoundError(
    #                 f"The default definition file {definition_file} does not "
    #                 " exist. Pass explicitly the "
    #                 "definition file as argument `definition_file`."
    #             )
    #     elif not definition_file.exists():
    #         raise FileNotFoundError(
    #             f"The file {definition_file} does not " "exist"
    #         )
    #     definition = _parse_def_file(definition_file)
    #     da = xr.open_dataset(path, chunks="auto")
    #     return definition.dataset_class(da, definition)
