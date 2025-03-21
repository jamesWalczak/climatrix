from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from climatrix.decorators import raise_if_not_installed
from climatrix.reconstruct.base import BaseReconstructor

if TYPE_CHECKING:
    from climatrix.dataset.sparse import SparseDataset


class SirenINRReconstructor(BaseReconstructor):

    def __init__(
        self,
        dataset: SparseDataset,
        lat: slice | np.ndarray = slice(-90, 90, 0.1),
        lon: slice | np.ndarray = slice(-180, 180, 0.1),
    ):
        super().__init__(dataset, lat, lon)
        if self.dataset.is_dynamic:
            raise ValueError(
                "Siren INR is not supported for dynamic datasets."
            )

    @raise_if_not_installed("...")
    def reconstruct(self):
        from climatrix.dataset.dense import StaticDenseDataset

        return StaticDenseDataset(
            xr.DataArray(
                values,
                coords=coords,
                dims=(self.dataset.latitude_name, self.dataset.longitude_name),
                name=self.dataset.da.name,
            ),
        )
