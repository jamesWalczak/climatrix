from __future__ import annotations

__all__ = ("StaticDenseDataset", "DynamicDenseDataset")
import warnings
from typing import TYPE_CHECKING, Self

import numpy as np
import xarray as xr
from matplotlib.axes import Axes

from climatrix.dataset.base import BaseDataset
from climatrix.decorators import raise_if_not_installed
from climatrix.exceptions import DatasetCreationError
from climatrix.sampling.base import NaNPolicy
from climatrix.sampling.type import SamplingType

if TYPE_CHECKING:
    from climatrix.dataset.models import DatasetDefinition
    from climatrix.dataset.sparse import SparseDataset


class DenseDataset(BaseDataset):

    def __new__(
        cls, da: xr.Dataset | xr.DataArray, definition: DatasetDefinition
    ) -> Self:
        if cls is DenseDataset:
            if definition.time_name is not None:
                return DynamicDenseDataset(da, definition)
            return StaticDenseDataset(da, definition)
        return super().__new__(cls)

    def maybe_roll(self, indexers: dict[str, slice]) -> Self:

        # Code derived from https://github.com/CMCC-Foundation/geokube
        first_el, last_el = (
            self.longitude.values.min(),
            self.longitude.values.max(),
        )
        start = indexers[self._def.longitude_name].start
        stop = indexers[self._def.longitude_name].stop
        start = 0 if start is None else start
        stop = 0 if stop is None else stop
        sel_neg_conv = (start < 0) | (stop < 0)
        sel_pos_conv = (start > 180) | (stop > 180)

        dset_neg_conv = first_el < 0
        dset_pos_conv = first_el >= 0

        if dset_pos_conv and sel_neg_conv:
            roll_value = (
                (self.da[self._def.longitude_name] >= 180).sum().item()
            )
            res = self.da.assign_coords(
                {
                    self._def.longitude_name: (
                        ((self.da[self._def.longitude_name] + 180) % 360) - 180
                    )
                }
            ).roll(**{self._def.longitude_name: roll_value}, roll_coords=True)
            res[self._def.longitude_name].attrs.update(
                self.da[self._def.longitude_name].attrs
            )
            return DenseDataset(res, self._def)
        if dset_neg_conv and sel_pos_conv:
            roll_value = (self.da[self._def.longitude_name] <= 0).sum().item()
            res = (
                self.da.assign_coords(
                    {
                        self._def.longitude_name: (
                            self.da[self._def.longitude_name] % 360
                        )
                    }
                )
                .roll(
                    **{self._def.longitude_name: -roll_value}, roll_coords=True
                )
                .assign_attrs(**self.da[self._def.longitude_name].attrs)
            )
            res[self._def.longitude_name].attrs.update(
                self.da[self._def.longitude_name].attrs
            )
            return DenseDataset(res, self._def)
        return DenseDataset(self.da, self._def)

    def subset(
        self,
        north: float | None = None,
        south: float | None = None,
        west: float | None = None,
        east: float | None = None,
    ) -> Self:
        if not (north or south or west or east):
            warnings.warn(
                "Subset parameters not provided. Returning the source dataset"
            )
            return DenseDataset(self.da, self._def)
        if north and south and north < south:
            raise ValueError("North must be greater than south")
        if west and east and west > east:
            raise ValueError("East must be greater than west")

        lats = self.latitude.values
        lons = self.longitude.values
        idx = {
            self._def.latitude_name: (
                np.s_[south:north]
                if np.all(np.diff(lats) >= 0)
                else np.s_[north:south]
            ),
            self._def.longitude_name: (
                np.s_[west:east]
                if np.all(np.diff(lons) >= 0)
                else np.s_[east:west]
            ),
        }
        da = self.maybe_roll(idx)
        da.da = da.da.sel(**idx)
        return da

    @classmethod
    def validate(self) -> None:
        # TODO: verify it is dense xarray dataset (gridded one)
        pass

    def sample(
        self,
        portion: float | None = None,
        number: int | None = None,
        *,
        kind: SamplingType | str = "uniform",
        nan_policy: NaNPolicy | str = "ignore",
        **sampler_kwargs,
    ) -> SparseDataset:
        return (
            SamplingType.get(kind)
            .value(self, portion=portion, number=number, **sampler_kwargs)
            .sample(nan_policy=nan_policy)
        )

    @raise_if_not_installed("hvplot", "panel")
    def plot(self, ax: Axes | None = None, **kwargs):
        from .plot import InteractiveDensePlotter

        InteractiveDensePlotter(self, **kwargs).show()


class StaticDenseDataset(DenseDataset):

    @classmethod
    def validate(self) -> None:
        super().validate()
        try:
            self.time
        except AttributeError:
            pass
        else:
            raise DatasetCreationError(
                "Static dense dataset must have no time dimension"
            )


class DynamicDenseDataset(DenseDataset):

    @classmethod
    def validate(self) -> None:
        super().validate()
        try:
            self.time
        except AttributeError:
            raise DatasetCreationError(
                "Dynamic dense dataset must have time dimension"
            )
