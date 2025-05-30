from __future__ import annotations

import logging

import numpy as np
from scipy.spatial import cKDTree

from climatrix.dataset.base import AxisType, BaseClimatrixDataset
from climatrix.dataset.domain import Domain
from climatrix.decorators.runtime import log_input
from climatrix.reconstruct.base import BaseReconstructor

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
    dataset : BaseClimatrixDataset
        The input dataset to reconstruct.
    target_domain : Domain
        The target domain for reconstruction.
    power : int, optional
        The power to raise the distance to (default is 2).
    k : int, optional
        The number of nearest neighbors to consider (default is 5).
    k_min : int, optional
        The minimum number of nearest neighbors to consider (if k < k_min)
        NaN values will be put (default is 2).

    Raises
    ------
    NotImplementedError
        If the input dataset is dynamic, as IDW reconstruction is not yet
        supported for dynamic datasets.
    ValueError
        If k_min is greater than k or if k is less than 1.
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
        for axis in dataset.domain.all_axes_types:
            if not dataset.domain.get_axis(axis).is_dimension:
                continue
            if axis in [AxisType.LATITUDE, AxisType.LONGITUDE, AxisType.POINT]:
                continue
            elif dataset.domain.get_size(axis) == 1:
                continue
            log.error(
                "Currently, IDWReconstructor only supports datasets with "
                "latitude and longitude dimensions, but got '%s'",
                axis,
            )
            raise NotImplementedError(
                "Currently, IDWReconstructor only supports datasets with "
                f"latitude and longitude dimensions, but got '{axis}'"
            )
        if k_min > k:
            log.error("k_min must be <= k")
            raise ValueError("k_min must be <= k")
        if k < 1:
            log.error("k must be >= 1")
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
        BaseClimatrixDataset
            The reconstructed dataset on the target domain.

        Notes
        -----
        - If fewer than `self.k_min` neighbors are available,
        NaN values are assigned to the corresponding points in the output.
        """
        values = self.dataset.da.values.flatten().squeeze()

        log.debug("Building KDtree for efficient nearest neighbor queries...")
        spatial_points = self.dataset.domain.get_all_spatial_points()
        if (
            not isinstance(spatial_points, np.ndarray)
            or spatial_points.ndim != 2
            or spatial_points.shape[1] != 2
        ):
            log.error(
                "Expected a 2D NumPy array with shape (N, 2) from "
                "get_all_spatial_points(), but got %s with shape %s.",
                type(spatial_points),
                getattr(spatial_points, "shape", None),
            )
            raise ValueError(
                "Expected a 2D NumPy array with shape (N, 2) from "
                f"get_all_spatial_points(), but got {type(spatial_points)} "
                f"with shape {getattr(spatial_points, 'shape', None)}."
            )
        kdtree = cKDTree(spatial_points)
        query_points = self.target_domain.get_all_spatial_points()
        log.debug("Querying %d nearest neighbors...", self.k)
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

        log.debug("Invalidating points with insufficient neighbors...")
        valid_neighbor_counts = np.isfinite(knn_data).sum(axis=1)
        interp_vals[valid_neighbor_counts < self.k_min] = np.nan

        log.debug("Reconstruction completed.")
        return BaseClimatrixDataset(
            self.target_domain.to_xarray(interp_vals, self.dataset.da.name)
        )
