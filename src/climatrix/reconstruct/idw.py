from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from scipy.spatial import cKDTree

from climatrix.decorators.runtime import log_input
from climatrix.reconstruct.base import BaseReconstructor

if TYPE_CHECKING:
    from climatrix.dataset.dense import DenseDataset
    from climatrix.dataset.sparse import SparseDataset

log = logging.getLogger(__name__)


class IDWReconstructor(BaseReconstructor):
    """
    Inverse Distance Weighting Reconstructor

    Attributes
    ----------
    k : int
        The number of nearest neighbors to consider.
    k_min : int
        The minimum number of nearest neighbors to consider (if k < k_min)
        NaN values will be put.
    power : int
        The power to raise the distance to

    Parameters
    ----------
    dataset : SparseDataset
        The input dataset.
    lat : slice or np.ndarray, optional
        The latitude range (default is slice(-90, 90, 0.1)).
    lon : slice or np.ndarray, optional
        The longitude range (default is slice(-180, 180, 0.1)).
    power : int, optional
        The power to raise the distance to (default is 2).
    k : int, optional
        The number of nearest neighbors to consider (default is 5).
    k_min : int, optional
        The minimum number of nearest neighbors to consider (if k < k_min)
        NaN values will be put (default is 2).
    """

    @log_input(log, level=logging.DEBUG)
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

    def _build_grid(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Build a grid of latitudes and longitudes.

        Parameters
        ----------
        self : IDWReconstructor

        Returns
        -------
        lat_grid : np.ndarray
            The latitude grid points.
        lon_grid : np.ndarray
            The longitude grid points.
        points : np.ndarray
            The input points in the form of a 2D array where each row is a point
            in the form (longitude, latitude).
        query_points : np.ndarray
            The query points in the form of a 2D array where each row is a point
            in the form (longitude, latitude) that will be used to query the
            nearest neighbors.
        """
        if isinstance(self.query_lat, slice):
            log.debug(
                "Creating target latitude values from slice %s...",
                self.query_lat,
            )
            lat_grid = np.arange(
                self.query_lat.start,
                self.query_lat.stop + self.query_lat.step,
                self.query_lat.step,
            )
        else:
            lat_grid = self.query_lat
        if isinstance(self.query_lon, slice):
            log.debug(
                "Creating target longitude values from slice %s...",
                self.query_lon,
            )
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

    def reconstruct(self) -> DenseDataset:
        """
        Perform Inverse Distance Weighting (IDW) reconstruction.

        This method reconstructs the sparse dataset using IDW,
        taking into account the specified number of nearest neighbors
        and the power to which distances are raised.
        The reconstructed data is returned as a dense dataset,
        either static or dynamic based on the input dataset.

        Returns
        -------
        DenseDataset
            The reconstructed dense dataset, either StaticDenseDataset
            or DynamicDenseDataset, depending on whether the input
            dataset is dynamic.

        Notes
        -----
        - For dynamic datasets, the reconstruction is performed for each
        time step.
        - If fewer than `self.k_min` neighbors are available,
        NaN values are assigned to the corresponding points in the output.
        """
        from climatrix.dataset.sparse import DynamicSparseDataset

        values = self.dataset.da.values

        lat_grid, lon_grid, points, query_points = self._build_grid()
        log.debug("Building KDtree for efficient nearest neighbor queries...")
        kdtree = cKDTree(points)
        log.debug("Querying %d nearest neighbors...", self.k)
        dists, idxs = kdtree.query(query_points, k=self.k, workers=-1)
        if self.k == 1:
            idxs = idxs[..., np.newaxis]
            dists = dists[..., np.newaxis]
        dists = np.maximum(dists, 1e-10)
        weights = 1 / np.power(dists, self.power)
        weights /= np.nansum(weights, axis=1, keepdims=True)

        if self.dataset.is_dynamic:
            from climatrix.dataset.dense import DynamicDenseDataset

            log.info("Reconstructing dynamic dataset...")
            time_values = []
            for t in range(len(self.dataset.time)):
                log.debug("Reconstructing time step: %s...", t)
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

            log.info("Reconstructing static dataset...")
            interp_vals = np.nansum(values[idxs] * weights, axis=1)
            interp_vals[np.isfinite(weights).sum(axis=1) < self.k_min] = (
                np.nan
            )            
            interp_vals = interp_vals.reshape(len(lat_grid), len(lon_grid))
            coords = {
                self.dataset.latitude_name: lat_grid,
                self.dataset.longitude_name: lon_grid,
            }
            dims = (
                self.dataset.latitude_name,
                self.dataset.longitude_name,
            )

            return StaticDenseDataset(
                xr.DataArray(
                    interp_vals,
                    coords=coords,
                    dims=dims,
                    name=self.dataset.da.name,
                ),
            )
