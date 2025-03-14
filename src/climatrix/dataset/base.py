from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from matplotlib.axes import Axes

from climatrix.dataset.models import DatasetDefinition
from climatrix.io import _parse_def_file


class BaseDataset(ABC):
    __slots__ = ("da", "def")

    da: xr.DataArray
    _def: DatasetDefinition

    def __init__(
        self, da: xr.Dataset | xr.DataArray, definition: DatasetDefinition
    ) -> None:
        if not isinstance(definition, DatasetDefinition):
            raise TypeError(
                "Definition can be created only based on "
                "DatasetDefinition object, "
                f"but provided {type(definition).__name__}"
            )
        if isinstance(da, xr.Dataset):
            if len(da.data_vars) > 1:
                raise ValueError(
                    "Dataset can be created only based on "
                    "xarray.DataArray or xarray.Dataset with single variable "
                    "objects, but provided xarray.Dataset with multiple "
                    "data_vars."
                )
            da = da[list(da.data_vars.keys())[0]]
        if not isinstance(da, xr.DataArray):
            raise TypeError(
                "Dataset can be created only based on "
                "xarray.DataArray or single-variable xarray.Dataset objects, "
                f"but provided {type(da).__name__}"
            )
        self.da = da
        self._def = definition
        self.validate()

    @property
    def latitude(self) -> xr.DataArray:
        return self.da[self._def.latitude_name]

    @property
    def longitude(self) -> xr.DataArray:
        return self.da[self._def.longitude_name]

    @property
    def time(self) -> xr.DataArray:
        if self._def.time_name is None:
            raise AttributeError(
                f"The dataset {self._def.name} has no time dimension"
            )
        return self.da[self._def.time_name]

    @abstractmethod
    def validate(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def plot(self, ax: Axes | None = None, **kwargs) -> Axes:
        raise NotImplementedError

    @classmethod
    def load(
        cls,
        path: str | os.PathLike | Path,
        definition_file: str | os.PathLike | Path | None = None,
    ) -> Self:
        path = Path(path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"The file {path} does not exist")
        if definition_file is None:
            definition_file: Path = path.with_suffix(path.suffix + ".def")
            if not definition_file.exists():
                raise FileNotFoundError(
                    f"The default definition file {definition_file} does not "
                    " exist. Pass explicitly the "
                    "definition file as argument `definition_file`."
                )
        elif not definition_file.exists():
            raise FileNotFoundError(
                f"The file {definition_file} does not " "exist"
            )
        definition = _parse_def_file(definition_file)
        da = xr.open_dataset(path, chunks="auto")
        return definition.dataset_class(da, definition)
