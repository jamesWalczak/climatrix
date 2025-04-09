from __future__ import annotations

__all__ = ("StaticSparseDataset", "DynamicSparseDataset")
import os
from pathlib import Path
from typing import TYPE_CHECKING

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.axes import Axes

from climatrix.dataset.base import BaseClimatrixDataset
from climatrix.reconstruct.type import ReconstructionType

if TYPE_CHECKING:
    from climatrix.dataset.dense import DenseDataset
    from climatrix.dataset.domain import Domain

SPARSE_DIM: str = "point"


class SparseDataset(BaseClimatrixDataset):

    def reconstruct(
        self,
        target: Domain,
        method: ReconstructionType | str = "idw",
        **recon_kwargs,
    ) -> DenseDataset:
        return (
            ReconstructionType.get(method)
            .value(
                self, lat=target.latitude, lon=target.longitude, **recon_kwargs
            )
            .reconstruct()
        )


class DynamicSparseDataset(SparseDataset):

    def __init__(self, xarray_obj: xr.DataArray):
        super().__init__(xarray_obj)
        raise NotImplementedError(
            "DynamicSparseDataset is not " "implemented yet"
        )

    def plot(
        self,
        target: str | os.PathLike | Path | None = None,
        show: bool = False,
        **kwargs,
    ) -> None:
        raise NotImplementedError(
            "Plotting of sparse datasets is not implemented yet"
        )


class StaticSparseDataset(SparseDataset):

    def __init__(self, xarray_obj: xr.DataArray):
        super().__init__(xarray_obj)

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
        size = kwargs.pop("size", 10)
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

        scatter = ax.scatter(
            lon,
            lat,
            c=self.da,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            transform=proj,
            marker="x",
            s=size,
        )

        cbar = plt.colorbar(
            scatter, ax=ax, orientation="vertical", shrink=0.7, pad=0.05
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
