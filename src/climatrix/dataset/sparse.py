from __future__ import annotations

__all__ = ("StaticSparseDataset", "DynamicSparseDataset")
from typing import TYPE_CHECKING, Self

import xarray as xr

from climatrix.dataset.base import BaseDataset
from climatrix.dataset.models import DatasetDefinition
from climatrix.exceptions import DatasetCreationError

if TYPE_CHECKING:
    from climatrix.dataset.dense import DenseDataset


class SparseDataset(BaseDataset):

    def __new__(
        cls, dset: xr.Dataset | xr.DataArray, definition: DatasetDefinition
    ) -> Self:
        if definition.time_name is not None:
            return DynamicSparseDataset(dset, definition)
        return StaticSparseDataset(dset, definition)

    @classmethod
    def validate(cls, dset: xr.Dataset) -> None:
        # TODO: verify it is dense xarray dataset (gridded one)
        pass

    def reconstruct(self) -> DenseDataset: ...


class StaticSparseDataset(SparseDataset):

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


class DynamicSparseDataset(SparseDataset):
    @classmethod
    def validate(self) -> None:
        super().validate()
        try:
            self.time
        except AttributeError:
            raise DatasetCreationError(
                "Dynamic dense dataset must have time dimension"
            )
