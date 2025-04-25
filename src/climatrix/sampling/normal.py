from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from climatrix.dataset.axis import Axis
from climatrix.sampling.base import BaseSampler

if TYPE_CHECKING:
    from climatrix.dataset.dense import DenseDataset
    from climatrix.dataset.sparse import SparseDataset

Longitude = float
Latitude = float


class NormalSampler(BaseSampler):

    def __init__(
        self,
        dataset: DenseDataset,
        portion: float | None = None,
        number: int | None = None,
        center_point: tuple[Longitude, Latitude] = None,
        sigma: float = 10.0,
    ):
        super().__init__(dataset, portion, number)
        self.center_point = center_point
        self.sigma = sigma

    def _sample_data(self, n) -> SparseDataset:
        if self.center_point is None:
            center_point = np.array(
                [
                    np.mean(self.dataset.latitude),
                    np.mean(self.dataset.longitude),
                ]
            )
        else:
            center_point = np.array(self.center_point)

        x_grid, y_grid = np.meshgrid(
            self.dataset.longitude, self.dataset.latitude
        )
        distances = np.sqrt(
            (x_grid - center_point[0]) ** 2 + (y_grid - center_point[1]) ** 2
        )
        weights = np.exp(-(distances**2) / (2 * self.sigma**2))
        weights /= weights.sum()

        flat_x = x_grid.flatten()
        flat_y = y_grid.flatten()

        indices = np.random.choice(len(flat_x), size=n, p=weights.flatten())
        selected_lats = flat_y[indices]
        selected_lons = flat_x[indices]
        data = self.dataset.da.sel(
            {
                self.dataset.latitude_name: xr.DataArray(
                    selected_lats, dims=[Axis.POINT]
                ),
                self.dataset.longitude_name: xr.DataArray(
                    selected_lons, dims=[Axis.POINT]
                ),
            },
            method="nearest",
        )
        if self.dataset.is_dynamic:
            from climatrix.dataset.sparse import DynamicSparseDataset

            return DynamicSparseDataset(data)
        else:
            from climatrix.dataset.sparse import StaticSparseDataset
        return StaticSparseDataset(data)

    def _sample_no_nans(self, lats, lons, n) -> SparseDataset:
        raise NotImplementedError
