from __future__ import annotations

import logging

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


class SIRENDataset(Dataset):
    """
    Dataset for training SIREN on 3D point clouds.

    This class prepares data for implicit neural representation
    learning by processing coordinate-value pairs and creating on-surface
    and off-surface points. The dataset normalizes coordinates and calculates
    surface normals for better learning.
    """

    def __init__(
        self,
        coordinates: np.ndarray,
        values: np.ndarray,
        on_surface_points: int,
    ) -> None:
        """
        Initialize the SIREN dataset with coordinates and corresponding values.

        Parameters
        ----------
        coordinates : np.ndarray
            Array of shape [N, 2] containing the coordinate points
        values : np.ndarray
            Array of shape [N] containing the values at each coordinate
        on_surface_points : int
            Number of on-surface points to sample per batch

        Raises
        ------
        ValueError
            If the number of coordinates doesn't match the number of values
        """
        if len(values.shape) == 1:
            values = values.reshape(-1, 1)
        if coordinates.shape[0] != values.shape[0]:
            log.error(
                f"Mismatch between coordinates ({coordinates.shape[0]})"
                f" and values ({values.shape[0]}) count"
            )
            raise ValueError(
                f"Mismatch between coordinates ({coordinates.shape[0]})"
                f" and values ({values.shape[0]}) count"
            )

        if coordinates.shape[0] < on_surface_points:
            log.warning(
                f"Only {coordinates.shape[0]} points available, "
                f"but requested {on_surface_points} on-surface points. "
                f"Reducing on_surface_points to match available data."
            )
            self.on_surface_points = coordinates.shape[0]
        else:
            self.on_surface_points = on_surface_points

        points_3d = np.concatenate([coordinates, values], axis=1)
        log.info(
            f"Created 3D points from coordinates and values: {points_3d.shape}"
        )

        self.coord_min = np.min(points_3d, axis=0)
        self.coord_max = np.max(points_3d, axis=0)

        normals_np = self._calculate_normals(points_3d)

        self.points_3d = torch.tensor(points_3d, dtype=torch.float32)
        self.normals = torch.tensor(normals_np, dtype=torch.float32)

        if self.normals.shape != self.points_3d.shape:
            log.error(
                f"Mismatch between points ({self.points_3d.shape})"
                f" and calculated normals ({self.normals.shape})"
                f" shape. Setting normals to zeros."
            )
            self.normals = torch.zeros_like(self.points_3d)

        self.points_3d = self._normalize_coords(self.points_3d)
        self.points_3d = torch.tensor(self.points_3d, dtype=torch.float32)
        log.info("Normalized 3D on-surface points.")

        self.total_points = self.points_3d.shape[0]

        log.info(
            f"Created dataset with {self.on_surface_points}"
            f" total on-surface points"
        )

    def _calculate_normals(self, points: np.ndarray) -> np.ndarray:
        """
        Calculate surface normals from point cloud data.

        Parameters
        ----------
        points : np.ndarray
            Array of shape [N, 3] containing 3D points

        Returns
        -------
        np.ndarray
            Array of shape [N, 3] containing the normal vectors
        """
        k_neighbors = min(30, points.shape[0] - 1)

        nnbrs = NearestNeighbors(
            n_neighbors=k_neighbors + 1, algorithm="auto"
        ).fit(points)
        distances, indices = nnbrs.kneighbors(points)

        normals = np.zeros_like(points)
        for i in range(points.shape[0]):
            neighbors = points[indices[i, 1:]]

            neighbors = neighbors - np.mean(neighbors, axis=0)

            pca = PCA(n_components=3)
            pca.fit(neighbors)

            normals[i] = pca.components_[2]

        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.maximum(norms, 1e-10)

        return normals

    def _normalize_coords(self, coords_3d: torch.Tensor) -> torch.Tensor:
        """
        Normalize coordinates to the range [-1, 1].

        Parameters
        ----------
        coords_3d : torch.Tensor
            Containing 3D coordinates

        Returns
        -------
        torch.Tensor
            Containing normalized coordinates
        """
        normalized = (coords_3d - self.coord_min) / (
            self.coord_max - self.coord_min
        )
        normalized = normalized * 2.0 - 1.0
        return normalized

    def __len__(self) -> int:
        """
        Return the number of batches in the dataset.

        Returns
        -------
        int
            Number of available batches
        """
        return self.total_points // self.on_surface_points

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return a batch of mixed on-surface and off-surface points.

        Parameters
        ----------
        idx : int
            Index of the batch to retrieve (not used in this
            implementation as points are randomly sampled)

        Returns
        -------
        tuple
            Contains:
            - coords : torch.Tensor
                Containing coordinates
            - sdf : torch.Tensor
                Containing signed distance values
            - normals : torch.Tensor
                Containing normal vectors
        """
        on_surface_samples = self.on_surface_points
        off_surface_samples = self.on_surface_points

        rand_idcs = torch.randint(0, self.total_points, (on_surface_samples,))
        on_surface_coords = self.points_3d[rand_idcs]
        on_surface_normals = self.normals[rand_idcs]

        off_surface_coords = torch.FloatTensor(
            off_surface_samples, 3
        ).uniform_(-1, 1)

        off_surface_normals = (
            torch.ones((off_surface_samples, 3), dtype=torch.float32) * -1
        )

        on_surface_sdf = torch.zeros(
            (on_surface_samples, 1), dtype=torch.float32
        )
        off_surface_sdf = (
            torch.ones((off_surface_samples, 1), dtype=torch.float32) * -1
        )

        coords = torch.cat([on_surface_coords, off_surface_coords], dim=0)
        normals = torch.cat([on_surface_normals, off_surface_normals], dim=0)
        sdf = torch.cat([on_surface_sdf, off_surface_sdf], dim=0)

        shuffle_idx = torch.randperm(coords.shape[0])

        return (
            coords[shuffle_idx],
            sdf[shuffle_idx],
            normals[shuffle_idx],
        )
