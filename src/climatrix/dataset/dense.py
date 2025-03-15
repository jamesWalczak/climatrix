from __future__ import annotations

from climatrix.dataset.axis import Axis
from climatrix.dataset.sparse import SparseDataset

__all__ = ("StaticDenseDataset", "DynamicDenseDataset")
import warnings
from typing import Self
from pathlib import Path
import os

import numpy as np
import xarray as xr
from matplotlib.axes import Axes

from climatrix.dataset.base import BaseClimatrixDataset
from climatrix.sampling.base import BaseSampler, NaNPolicy
from climatrix.sampling.type import SamplingType


class DenseDataset(BaseClimatrixDataset):

    def maybe_roll(self, indexers: dict[str, slice]) -> Self:
        # Code derived from https://github.com/CMCC-Foundation/geokube
        first_el, last_el = (
            self.longitude.values.min(),
            self.longitude.values.max(),
        )
        start = indexers[self.longitude_name].start
        stop = indexers[self.longitude_name].stop
        start = 0 if start is None else start
        stop = 0 if stop is None else stop
        sel_neg_conv = (start < 0) | (stop < 0)
        sel_pos_conv = (start > 180) | (stop > 180)

        dset_neg_conv = first_el < 0
        dset_pos_conv = first_el >= 0

        res = self.da
        if dset_pos_conv and sel_neg_conv:
            roll_value = (
                (self.da[self.longitude_name] >= 180).sum().item()
            )
            res = self.da.assign_coords(
                {
                    self.longitude_name: (
                        ((self.da[self.longitude_name] + 180) % 360) - 180
                    )
                }
            ).roll(**{self.longitude_name: roll_value}, roll_coords=True)
            res[self.longitude_name].attrs.update(
                self.da[self.longitude_name].attrs
            )
        if dset_neg_conv and sel_pos_conv:
            roll_value = (self.da[self.longitude_name] <= 0).sum().item()
            res = (
                self.da.assign_coords(
                    {
                        self.longitude_name: (
                            self.da[self.longitude_name] % 360
                        )
                    }
                )
                .roll(
                    **{self.longitude_name: -roll_value}, roll_coords=True
                )
                .assign_attrs(**self.da[self.longitude_name].attrs)
            )
            res[self.longitude_name].attrs.update(
                self.da[self.longitude_name].attrs
            )
        return type(self)(res)
    
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
            return type(self)(self.da)
        if north and south and north < south:
            raise ValueError("North must be greater than south")
        if west and east and west > east:
            raise ValueError("East must be greater than west")

        lats = self.latitude.values
        lons = self.longitude.values
        idx = {
            self.latitude_name: (
                np.s_[south:north]
                if np.all(np.diff(lats) >= 0)
                else np.s_[north:south]
            ),
            self.longitude_name: (
                np.s_[west:east]
                if np.all(np.diff(lons) >= 0)
                else np.s_[east:west]
            ),
        }
        da = self.maybe_roll(idx)
        da.da = da.da.sel(**idx)
        return da    
    
    def sample(
        self,
        portion: float | None = None,
        number: int | None = None,
        *,
        kind: SamplingType | str = "uniform",
        nan_policy: NaNPolicy | str = "ignore",
        **sampler_kwargs,
    ) -> SparseDataset:
        class_: type[BaseSampler] = SamplingType.get(kind).value
        return (class_(self, portion=portion, number=number, **sampler_kwargs)
            .sample(nan_policy=nan_policy)
        )    

class DynamicDenseDataset(DenseDataset):

    def plot(self, target: str | os.PathLike | Path | None = None, show: bool = False, **kwargs) -> None:
        raise NotImplementedError("Plotting of dynamic dense datasets is not implemented yet")

    

class StaticDenseDataset(DenseDataset):

    def plot(self, target: str | os.PathLike | Path | None = None, show: bool = False, **kwargs) -> None:
        if target is not None:
            target = Path(target)
            target.parent.mkdir(parents=True, exist_ok=True)
        # TODO: finish
        

class A:

    @classmethod
    def validate(self) -> None:
        # TODO: verify it is dense xarray dataset (gridded one)
        pass



    def plot(self, ax: Axes | None = None, **kwargs):
        from .plot import InteractiveDensePlotter

        InteractiveDensePlotter(self, **kwargs).show()


