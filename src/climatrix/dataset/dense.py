from __future__ import annotations

__all__ = ("StaticDenseDataset", "DynamicDenseDataset")
from typing import TYPE_CHECKING, Self

import xarray as xr
from matplotlib.axes import Axes

from climatrix.dataset.base import BaseDataset
from climatrix.exceptions import DatasetCreationError

if TYPE_CHECKING:
    from climatrix.dataset.models import DatasetDefinition
    from climatrix.dataset.sparse import SparseDataset


class DenseDataset(BaseDataset):

    def __new__(
        cls, dset: xr.Dataset | xr.DataArray, definition: DatasetDefinition
    ) -> Self:
        if cls is DenseDataset:
            if definition.time_name is not None:
                return DynamicDenseDataset(dset, definition)
            return StaticDenseDataset(dset, definition)
        return super().__new__(cls)

    @classmethod
    def validate(self) -> None:
        # TODO: verify it is dense xarray dataset (gridded one)
        pass

    def sample(self) -> SparseDataset: ...


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
