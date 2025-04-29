from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from scipy.spatial import cKDTree

from climatrix.dataset.domain import Domain
from climatrix.decorators.runtime import log_input
from climatrix.reconstruct.base import BaseReconstructor

if TYPE_CHECKING:
    from climatrix.dataset.base import BaseClimatrixDataset

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
        dataset: BaseClimatrixDataset,
        target_domain: Domain,
        power: int = 2,
        k: int = 5,
        k_min: int = 2,
    ):
        super().__init__(dataset, target_domain)
        if dataset.domain.is_dynamic:
            raise ValueError(
                "IDW reconstruction is not yet supported for dynamic datasets."
            )
        if k_min > k:
            raise ValueError("k_min must be <= k")
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k
        self.k_min = k_min
        self.power = power

    def reconstruct(self) -> BaseClimatrixDataset:
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
        from climatrix.dataset.base import BaseClimatrixDataset

        values = self.dataset.da.values

        log.debug("Building KDtree for efficient nearest neighbor queries...")
        kdtree = cKDTree(self.dataset.domain.get_all_spatial_points())
        log.debug("Querying %d nearest neighbors...", self.k)
        query_points = self.target_domain.get_all_spatial_points()
        dists, idxs = kdtree.query(query_points, k=self.k, workers=-1)

        if self.k == 1:
            idxs = idxs[..., np.newaxis]
            dists = dists[..., np.newaxis]
        dists = np.maximum(dists, 1e-10)
        weights = 1.0 / np.power(dists, self.power)
        weights /= np.nansum(weights, axis=1, keepdims=True)

        knn_data = values[idxs]
        valid_mask = np.isfinite(knn_data)
        weights[~valid_mask] = 0.0
        weights_sum = np.nansum(weights, axis=1).squeeze()
        interp_vals = np.divide(
            np.nansum(knn_data * weights, axis=1),
            weights_sum,
            where=weights_sum != 0,
        )

        valid_neighbor_counts = np.isfinite(knn_data).sum(axis=1)
        interp_vals[valid_neighbor_counts < self.k_min] = np.nan

        if self.target_domain.is_sparse:
            coords = {
                self.target_domain.latitude_name: (
                    self.target_domain.point_name,
                    self.target_domain.latitude,
                ),
                self.target_domain.longitude_name: (
                    self.target_domain.point_name,
                    self.target_domain.longitude,
                ),
            }
            dims = (self.target_domain.point_name,)
            dset = xr.DataArray(
                interp_vals,
                coords=coords,
                dims=dims,
                name=self.dataset.da.name,
            )
        else:
            interp_vals = interp_vals.reshape(
                len(self.target_domain.latitude),
                len(self.target_domain.longitude),
            )
            coords = {
                self.target_domain.latitude_name: self.target_domain.latitude,
                self.target_domain.longitude_name: (
                    self.target_domain.longitude
                ),
            }
            dims = (
                self.target_domain.latitude_name,
                self.target_domain.longitude_name,
            )
            dset = xr.DataArray(
                interp_vals,
                coords=coords,
                dims=dims,
                name=self.dataset.da.name,
            )

        return BaseClimatrixDataset(dset)
