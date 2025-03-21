from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
import xarray as xr
from scipy.spatial import cKDTree

from climatrix.reconstruct.base import BaseReconstructor

if TYPE_CHECKING:
    from climatrix.dataset.sparse import SparseDataset


class IDWReconstructor(BaseReconstructor):

    def __init__(
        self,
        dataset: SparseDataset,
        lat: slice | np.ndarray = slice(-90, 90, 0.1),
        lon: slice | np.ndarray = slice(-180, 180, 0.1),
        power: int = 2,
        k: int = 5,
        k_min: int = 2,
    ):
        super().__init__(dataset, lat, lon)
        self.k = k
        self.k_min = k_min
        self.power = power

    def _build_grid(self) -> np.ndarray:
        if isinstance(self.query_lat, slice):
            lat_grid = np.arange(
                self.query_lat.start,
                self.query_lat.stop + self.query_lat.step,
                self.query_lat.step,
            )
        else:
            lat_grid = self.query_lat
        if isinstance(self.query_lon, slice):
            lon_grid = np.arange(
                self.query_lon.start,
                self.query_lon.stop + self.query_lon.step,
                self.query_lon.step,
            )
        else:
            lon_grid = self.query_lon
        points = np.column_stack(
            (self.dataset.longitude.values, self.dataset.latitude.values)
        )
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
        return (
            lat_grid,
            lon_grid,
            points,
            np.column_stack((lon_mesh.flatten(), lat_mesh.flatten())),
        )

    def reconstruct(self):
        from climatrix.dataset.dense import DenseDataset
        from climatrix.dataset.sparse import DynamicSparseDataset

        values = self.dataset.da.values

        lat_grid, lon_grid, points, query_points = self._build_grid()
        kdtree = cKDTree(points)
        dists, idxs = kdtree.query(query_points, k=self.k, workers=-1)
        if self.k == 1:
            idxs = idxs[..., np.newaxis]
            dists = dists[..., np.newaxis]
        dists = np.maximum(dists, 1e-10)
        weights = 1 / np.power(dists, self.power)
        weights /= np.nansum(weights, axis=1, keepdims=True)

        if isinstance(self.dataset, DynamicSparseDataset):
            time_values = []
            for t in range(len(self.dataset.time)):
                v = values[t, :]
                interp_vals = np.nansum(v[idxs] * weights, axis=1)
                interp_vals[np.isfinite(weights).sum(axis=1) < self.k_min] = (
                    np.nan
                )
                time_values.append(
                    interp_vals.reshape(len(lat_grid), len(lon_grid))
                )
            interp_data = np.stack(time_values, axis=0)
            coords = {
                self.dataset.time_name: self.dataset.time.values,
                self.dataset.latitude_name: lat_grid,
                self.dataset.longitude_name: lon_grid,
            }
            dims = (
                self.dataset.time_name,
                self.dataset.latitude_name,
                self.dataset.longitude_name,
            )
        else:
            interp_data = np.nansum(values[idxs] * weights, axis=1)
            interp_data = interp_data.reshape(len(lat_grid), len(lon_grid))
            coords = {
                self.dataset.latitude_name: lat_grid,
                self.dataset.longitude_name: lon_grid,
            }
            dims = (
                self.dataset.latitude_name,
                self.dataset.longitude_name,
            )
        if self.dataset.is_dynamic:
            from climatrix.dataset.dense import DynamicDenseDataset

            return DynamicDenseDataset(
                xr.DataArray(
                    interp_data,
                    coords=coords,
                    dims=dims,
                    name=self.dataset.da.name,
                ),
            )
        else:
            from climatrix.dataset.dense import StaticDenseDataset

            return StaticDenseDataset(
                xr.DataArray(
                    interp_data,
                    coords=coords,
                    dims=dims,
                    name=self.dataset.da.name,
                ),
            )
