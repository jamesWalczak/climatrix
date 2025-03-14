from __future__ import annotations

__all__ = ("StaticSparseDataset", "DynamicSparseDataset")
from typing import TYPE_CHECKING, ClassVar, Self

import xarray as xr
from matplotlib.axes import Axes

from climatrix.dataset.base import BaseDataset
from climatrix.dataset.models import DatasetDefinition
from climatrix.decorators import raise_if_not_installed

if TYPE_CHECKING:
    from climatrix.dataset.dense import DenseDataset


class SparseDataset(BaseDataset):
    SPARSE_DIM: ClassVar[str] = "point"

    def __new__(
        cls, da: xr.Dataset | xr.DataArray, definition: DatasetDefinition
    ) -> Self:
        if cls is SparseDataset:
            if definition.time_name is not None:
                return DynamicSparseDataset(da, definition)
            return StaticSparseDataset(da, definition)
        return super().__new__(cls)

    def validate(self) -> None:
        # TODO: verify it is dense xarray dataset (gridded one)
        pass

    @raise_if_not_installed("hvplot", "panel")
    def plot(self, ax: Axes | None = None, **kwargs):
        from climatrix.dataset.plot import InteractiveScatterPlotter

        InteractiveScatterPlotter(self).show()

        # import cartopy.crs as ccrs
        # import cartopy.feature as cfeature

        # projection = ccrs.PlateCarree()  # Or another projection like ccrs.Robinson()

        # # Create the figure and axes
        # if ax is None:
        #     fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={'projection': projection})

        # # Add map features (coastlines, etc.)
        # breakpoint()
        # ax.coastlines()
        # ax.add_feature(cfeature.BORDERS, linestyle=':')
        # ax.gridlines(draw_labels=True, linestyle='--')

        # # Plot the scatter points
        # sc = ax.scatter(
        #     self.longitude.values,
        #     self.latitude.values,
        #     c=self.da[field].isel(valid_time=0).values,  # Color the points by temperature
        #     s=50,  # Adjust marker size
        #     transform=ccrs.PlateCarree(),  # Important: Data is in lat/lon
        #     cmap='viridis',  # Choose a colormap
        # )

        # # Add a colorbar
        # plt.colorbar(sc, ax=ax, label='Temperature')

        # # Set plot title
        # plt.title('Sparse Temperature Points')

        # # Show the plot
        # plt.show()

    def reconstruct(self) -> DenseDataset: ...


class StaticSparseDataset(SparseDataset): ...


class DynamicSparseDataset(SparseDataset): ...
