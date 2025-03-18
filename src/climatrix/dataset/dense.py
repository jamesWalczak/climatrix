from __future__ import annotations

from climatrix.dataset.axis import Axis

__all__ = ("StaticDenseDataset", "DynamicDenseDataset")
import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Self

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes

from climatrix.dataset.base import BaseClimatrixDataset
from climatrix.sampling.base import BaseSampler, NaNPolicy
from climatrix.sampling.type import SamplingType

if TYPE_CHECKING:
    from climatrix.dataset.sparse import SparseDataset


# matplotlib.use('Agg')
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
            roll_value = (self.da[self.longitude_name] >= 180).sum().item()
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
                    {self.longitude_name: self.da[self.longitude_name] % 360}
                )
                .roll(**{self.longitude_name: -roll_value}, roll_coords=True)
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
        return class_(
            self, portion=portion, number=number, **sampler_kwargs
        ).sample(nan_policy=nan_policy)


class DynamicDenseDataset(DenseDataset):
    def plot(
        self,
        title: str | None = None,
        target: str | os.PathLike | Path | None = None,
        show: bool = True,
        **kwargs,
    ) -> Axes:
        figsize = kwargs.pop("figsize", (12, 6))
        vmin = kwargs.pop("vmin", None)
        vmax = kwargs.pop("vmax", None)
        cmap = kwargs.pop("cmap", "seismic")
        ax = kwargs.pop("ax", None)
        cbar_name = kwargs.pop("cbar_name", None)
        title = title or self.da.name or "Climatrix Dataset"
        if (
            self.time_name in self.da.dims
            and self.da.sizes[self.time_name] > 1
        ):
            data_2d = self.da.isel({self.time_name: 0})
        else:
            data_2d = self.da

        lat = data_2d[self.latitude_name]
        lon = data_2d[self.longitude_name]

        proj = ccrs.PlateCarree()

        if ax is None:
            fig, ax = plt.subplots(
                figsize=figsize, subplot_kw={"projection": proj}
            )
            ax.coastlines()
            ax.add_feature(cfeature.BORDERS, linewidth=0.5)
            ax.add_feature(cfeature.LAND, facecolor="lightgray")
            ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
            ax.gridlines(draw_labels=True, linewidth=0.2, linestyle="--")

            ax.text(
                -0.07,
                0.55,
                "latitude",
                va="bottom",
                ha="center",
                rotation="vertical",
                rotation_mode="anchor",
                transform=ax.transAxes,
            )
            ax.text(
                0.5,
                -0.1,
                "longitude",
                va="bottom",
                ha="center",
                rotation="horizontal",
                rotation_mode="anchor",
                transform=ax.transAxes,
            )

            ax.set_aspect("equal")

        mesh = ax.pcolormesh(
            lon,
            lat,
            data_2d,
            transform=proj,
            cmap=cmap,
            shading="auto",
            vmin=vmin,
            vmax=vmax,
        )

        cbar = plt.colorbar(
            mesh, ax=ax, orientation="vertical", shrink=0.7, pad=0.05
        )
        cbar.set_label(cbar_name or data_2d.name or "Value")

        if title:
            ax.set_title(title, fontsize=14)

        plt.tight_layout()
        if target is not None:
            target = Path(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(target, dpi=300)
        if show:
            plt.show()
        return ax


class StaticDenseDataset(DenseDataset):
    def plot(
        self,
        title: str | None = None,
        target: str | os.PathLike | Path | None = None,
        show: bool = True,
        **kwargs,
    ) -> Axes:
        figsize = kwargs.pop("figsize", (12, 6))
        vmin = kwargs.pop("vmin", None)
        vmax = kwargs.pop("vmax", None)
        cmap = kwargs.pop("cmap", "seismic")
        ax = kwargs.pop("ax", None)
        cbar_name = kwargs.pop("cbar_name", None)
        title = title or self.da.name or "Climatrix Dataset"

        lat = self.da[self.latitude_name]
        lon = self.da[self.longitude_name]

        proj = ccrs.PlateCarree()

        if ax is None:
            fig, ax = plt.subplots(
                figsize=figsize, subplot_kw={"projection": proj}
            )
            ax.coastlines()
            ax.add_feature(cfeature.BORDERS, linewidth=0.5)
            ax.add_feature(cfeature.LAND, facecolor="lightgray")
            ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
            ax.gridlines(draw_labels=True, linewidth=0.2, linestyle="--")

            ax.text(
                -0.07,
                0.55,
                "latitude",
                va="bottom",
                ha="center",
                rotation="vertical",
                rotation_mode="anchor",
                transform=ax.transAxes,
            )
            ax.text(
                0.5,
                -0.1,
                "longitude",
                va="bottom",
                ha="center",
                rotation="horizontal",
                rotation_mode="anchor",
                transform=ax.transAxes,
            )

            ax.set_aspect("equal")

        mesh = ax.pcolormesh(
            lon,
            lat,
            self.da,
            transform=proj,
            cmap=cmap,
            shading="auto",
            vmin=vmin,
            vmax=vmax,
        )

        cbar = plt.colorbar(
            mesh, ax=ax, orientation="vertical", shrink=0.7, pad=0.05
        )
        cbar.set_label(cbar_name or self.da.name or "Value")

        if title:
            ax.set_title(title, fontsize=14)

        plt.tight_layout()
        if target is not None:
            target = Path(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(target, dpi=300)
        if show:
            plt.show()
        return ax
