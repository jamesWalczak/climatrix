from __future__ import annotations

import os
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Self

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from matplotlib.axes import Axes

from climatrix.dataset.models import DatasetDefinition
from climatrix.decorators import raise_if_not_installed
from climatrix.io import _parse_def_file


class BaseDataset(ABC):
    __slots__ = ("dset", "def")

    cube: xr.Dataset
    _def: DatasetDefinition

    def __init__(
        self, dset: xr.Dataset | xr.DataArray, definition: DatasetDefinition
    ) -> None:
        if not isinstance(definition, DatasetDefinition):
            raise TypeError(
                "Definition can be created only based on "
                "DatasetDefinition object, "
                f"but provided {type(definition).__name__}"
            )
        if isinstance(dset, xr.DataArray):
            dset = dset.to_dataset()
        if not isinstance(dset, (xr.Dataset)):
            raise TypeError(
                "Dataset can be created only based on "
                "xarray.DataArray or xarray.Dataset objects, "
                f"but provided {type(dset).__name__}"
            )
        self.dset = dset
        self._def = definition
        self.validate()

    @property
    def latitude(self) -> xr.DataArray:
        return self.dset[self._def.latitude_name]

    @property
    def longitude(self) -> xr.DataArray:
        return self.dset[self._def.longitude_name]

    @property
    def time(self) -> xr.DataArray:
        if self._def.time_name is None:
            raise AttributeError(
                f"The dataset {self._def.name} has no time dimension"
            )
        return self.dset[self._def.time_name]

    @property
    def fields_names(self) -> tuple[str]:
        return tuple(self.dset.data_vars)

    @abstractmethod
    def validate(self) -> None:
        raise NotImplementedError

    @raise_if_not_installed("hvplot", "panel")
    def plot(self, ax: Axes | None = None, **kwargs) -> Axes:
        from .plot import InteractiveDensePlotter

        # TODO: to remove
        res = self.sample(number=100).plot()

        InteractiveDensePlotter(self, **kwargs).show()

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
        dset = xr.open_dataset(path, chunks="auto")
        return definition.dataset_class(dset, definition)
