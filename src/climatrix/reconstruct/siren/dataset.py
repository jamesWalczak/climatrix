from __future__ import annotations

import logging
from collections import namedtuple
from typing import TYPE_CHECKING

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

from climatrix.decorators.runtime import log_input

if TYPE_CHECKING:
    from climatrix.dataset.sparse import SparseDataset

SdfEntry = namedtuple("SdfEntry", ["coords", "normals", "sdf"])

log = logging.getLogger(__name__)


class SDFPredictDataset(Dataset):
    @log_input(log, level=logging.DEBUG)
    def __init__(
        self,
        coords: np.ndarray,
        keep_aspect_ratio: bool = True,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        coords = self.center(coords)
        self.coords = self.normalize(
            coords, keep_aspect_ratio=keep_aspect_ratio
        )
        self.device = device or torch.device("cpu")

    def __len__(self) -> int:
        return len(self.coords)

    def center(self, coords: np.ndarray) -> np.ndarray:
        return coords - np.mean(coords, axis=0, keepdims=True)

    def normalize(
        self, coords: np.ndarray, keep_aspect_ratio: bool
    ) -> np.ndarray:
        if keep_aspect_ratio:
            self.coord_max = np.max(coords)
            self.coord_min = np.min(coords)
        else:
            self.coord_max = np.max(coords, axis=0, keepdims=True)
            self.coord_min = np.min(coords, axis=0, keepdims=True)

        # Normalize to [0, 1]
        coords = (coords - self.coord_min) / (self.coord_max - self.coord_min)

        coords -= 0.5  # Normalize to [-0.5, 0.5]
        coords *= 2.0  # Normalize to [-1, 1]
        return coords

    def __getitem__(self, index) -> torch.Tensor:
        return torch.from_numpy(self.coords[index]).float()


class SDFTrainDataset(SDFPredictDataset):

    @log_input(log, level=logging.DEBUG)
    def __init__(
        self,
        coords: np.ndarray,
        num_surface_points: int = 1_000,
        num_off_surface_points: int = 1_000,
        keep_aspect_ratio: bool = True,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(coords, keep_aspect_ratio, device)
        self.num_surface_points = num_surface_points
        self.num_off_surface_points = num_off_surface_points

    def __len__(self) -> int:
        return len(self.coords) // self.num_surface_points

    def _compute_normals(self, idx) -> np.ndarray:
        from sklearn.neighbors import KDTree

        tree = KDTree(self.coords)
        _, nearest = tree.query(self.coords[idx], k=10)
        clusters = torch.from_numpy(self.coords[nearest]).float()
        clusters -= clusters.mean(dim=1, keepdim=True)
        cov_matrices = torch.matmul(clusters.transpose(1, 2), clusters) / (
            clusters.shape[1] - 1
        )
        _, eigvecs = torch.linalg.eigh(cov_matrices)
        normal_vectors = eigvecs[:, :, 0]
        return normal_vectors

    def _sample_on_surface(self) -> tuple[np.ndarray, np.ndarray]:
        idx = np.random.choice(len(self.coords), size=self.num_surface_points)
        normals = self._compute_normals(idx)
        return self.coords[idx, :], normals

    def _sample_off_surface(self) -> tuple[np.ndarray, np.ndarray]:
        coords = np.random.uniform(
            -1, 1, size=(self.num_off_surface_points, 3)
        )
        normals = np.ones((self.num_off_surface_points, 3)) * -1
        return (coords, normals)

    def _sample_sdf(self) -> np.ndarray:
        total_points = self.num_surface_points + self.num_off_surface_points
        sdf = np.zeros((total_points, 1))
        # NOTE: off-surface are only external points. to consider internal
        sdf[self.num_surface_points :] = -1
        return sdf

    def __getitem__(self, index) -> SdfEntry:
        on_surface_coords, on_surface_normals = self._sample_on_surface()
        off_surface_coords, off_surface_normals = self._sample_off_surface()
        sdf = self._sample_sdf()

        coords = np.concatenate(
            (on_surface_coords, off_surface_coords), axis=0
        )
        normals = np.concatenate(
            (on_surface_normals, off_surface_normals), axis=0
        )
        return SdfEntry(
            torch.from_numpy(coords).float(),
            torch.from_numpy(normals).float(),
            torch.from_numpy(sdf).float(),
        )
