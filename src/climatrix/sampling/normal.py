from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from climatrix.dataset.sparse import (
    SPARSE_DIM,
    DynamicSparseDataset,
    StaticSparseDataset,
)
from climatrix.sampling.base import BaseSampler

if TYPE_CHECKING:
    from climatrix.dataset.dense import DenseDataset

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

    def _sample_data(self, lats, lons, n) -> SparseDataset:
        if self.center_point is None:
            center_point = np.array([np.mean(lats), np.mean(lons)])
        else:
            center_point = np.array(self.center_point)

        x_grid, y_grid = np.meshgrid(lons, lats)
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
                    selected_lats, dims=[SPARSE_DIM]
                ),
                self.dataset.longitude_name: xr.DataArray(
                    selected_lons, dims=[SPARSE_DIM]
                ),
            },
            method="nearest",
        )
        if self.dataset.is_dynamic:
            return DynamicSparseDataset(data)
        return StaticSparseDataset(data)

    def _sample_no_nans(self, lats, lons, n) -> SparseDataset:
        raise NotImplementedError
