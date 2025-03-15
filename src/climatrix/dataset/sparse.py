from __future__ import annotations

__all__ = ("StaticSparseDataset", "DynamicSparseDataset")
from typing import TYPE_CHECKING, ClassVar, Self

import xarray as xr
from matplotlib.axes import Axes

from climatrix.dataset.base import BaseClimatrixDataset
from climatrix.dataset.models import DatasetDefinition
from climatrix.decorators import raise_if_not_installed
from climatrix.reconstruct.type import ReconstructionType

if TYPE_CHECKING:
    from climatrix.dataset.dense import DenseDataset

class SparseDataset(BaseClimatrixDataset):
    SPARSE_DIM: ClassVar[str] = "point"
    
    def reconstruct(
        self,
        lat: slice,
        lon: slice,
        method: ReconstructionType | str = "idw",
        **recon_kwargs,
    ) -> DenseDataset:
        return (
            ReconstructionType.get(method)
            .value(self, lat=lat, lon=lon, **recon_kwargs)
            .reconstruct()
        )

class DynamicSparseDataset(SparseDataset):

    def __init__(self, xarray_obj: xr.DataArray):
        super().__init__(xarray_obj)
        raise NotImplementedError("DynamicSparseDataset is not implemented yet")
    

class StaticSparseDataset(SparseDataset):

    def __init__(self, xarray_obj: xr.DataArray):
        super().__init__(xarray_obj)







class A:
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



