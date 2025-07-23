from __future__ import annotations

import importlib.resources
import logging
from collections import namedtuple
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import torch
from scipy.spatial import cKDTree
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

from climatrix.decorators.runtime import log_input

if TYPE_CHECKING:
    pass

SdfEntry = namedtuple("SdfEntry", ["coordinates", "normals", "sdf"])

log = logging.getLogger(__name__)

_ELEVATION_DATASET_PATH: Path = importlib.resources.files(
    "climatrix.reconstruct.sinet"
).joinpath("resources", "lat_lon_elevation.npy")
_SLOPE_DATASET_PATH: Path = importlib.resources.files(
    "climatrix.reconstruct.sinet"
).joinpath("resources", "lat_lon_slope.npy")


def load_elevation_dataset() -> np.ndarray:
    """
    Load the elevation dataset.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple containing the coordinates and elevation data as numpy arrays.
        The first array contains the coordinates (latitude, longitude),
        and the second array contains the elevation values.
    """
    log.debug("Loading elevation dataset...")
    data = np.load(_ELEVATION_DATASET_PATH)
    return data[:, :-1], MinMaxScaler((-1, 1)).fit_transform(
        data[:, -1].reshape(-1, 1)
    )


def load_slope_dataset() -> np.ndarray:
    """
    Load the slope dataset.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple containing the coordinates and slope data as numpy arrays.
        The first array contains the coordinates (latitude, longitude),
        and the second array contains the slope values.
    """
    log.debug("Loading slope dataset...")
    data = np.load(_SLOPE_DATASET_PATH)
    return data[:, :-1], MinMaxScaler((-1, 1)).fit_transform(
        data[:, -1].reshape(-1, 1)
    )


def query_features(
    tree: cKDTree, values: np.ndarray, query_points: np.ndarray
):
    log.debug("Querying elevation and slope...")
    distances, indices = tree.query(query_points, k=1)
    if np.any(distances > 0.1):
        log.warning("Some coordinates are too far from the known data points.")
    return values[indices]


class SiNETDatasetGenerator:
    train_coordinates: np.ndarray
    train_field: np.ndarray
    target_coordinates: np.ndarray | None
    field_transformer: MinMaxScaler
    radius: float

    @log_input(log, level=logging.DEBUG)
    def __init__(
        self,
        spatial_points: np.ndarray,
        field: np.ndarray,
        *,
        target_coordinates: np.ndarray | None = None,
        degree: bool = True,
        radius: float = 1.0,
        val_portion: float = 0.2,
        use_elevation: bool = False,
        use_slope: bool = False,
    ) -> None:
        """
        Initialize a SiNET dataset generator.

        Parameters
        ----------
        spatial_points : np.ndarray
            Array on shape Nx2 with latitudes and longitudes of training points.
        field : np.ndarray
            Array of shape (N,) with field values at training points.
        target_coordinates : np.ndarray | None, optional
            Array on shape Nx2 with latitudes and longitudes of target points.
        degree : bool, optional
            Whether the input latitudes and longitudes are in degrees.
            Defaults to True.
        radius : float, optional
            The radius of the sphere. Defaults to 1.0.
        val_portion : float, optional
            Portion of the training data to use for validation. Defaults to 0.2.
        use_elevation: bool, optional
            Whether to use elevation data. Defaults to False.
        use_slope: bool, optional
            Whether to use slope data. Defaults to False.
        """
        if val_portion <= 0 or val_portion >= 1:
            log.error("Validation portion must be in the range (0, 1).")
            raise ValueError("Validation portion must be in the range (0, 1).")
        # NOTE: we expect slope and elevation to be defined

        self.field_transformer = MinMaxScaler((-1, 1))
        field = self.field_transformer.fit_transform(field.reshape(-1, 1))
        self.target_coordinates = target_coordinates
        if degree:
            log.debug("Converting degrees to radians...")
            spatial_points = np.deg2rad(spatial_points)
            if target_coordinates is not None:
                self.target_coordinates = np.deg2rad(target_coordinates)
        (
            self.train_coordinates,
            self.train_field,
            self.val_coordinates,
            self.val_field,
        ) = self._split_train_val(spatial_points, field, val_portion)

        ckdtree = None
        self.radius = radius
        if use_elevation:
            coords, self.elevation = load_elevation_dataset()
            ckdtree = cKDTree(np.deg2rad(coords))
            self._extend_input_featuers(ckdtree, self.elevation)
        if use_slope:
            coords, self.slope = load_slope_dataset()
            if ckdtree is None:
                ckdtree = cKDTree(np.deg2rad(coords))
            self._extend_input_featuers(ckdtree, self.slope)

    @property
    def n_features(self) -> int:
        """
        Number of features in the dataset.

        Returns
        -------
        int
            Number of features.
        """
        return self.train_coordinates.shape[1]

    def _extend_input_featuers(
        self, tree: cKDTree, values: np.ndarray
    ) -> np.ndarray:
        train_extra_feature = query_features(
            tree, values, self.train_coordinates[:, :2]
        )
        self.train_coordinates = np.concatenate(
            [self.train_coordinates, train_extra_feature.reshape(-1, 1)],
            axis=1,
        )

        val_extra_feature = query_features(
            tree, values, self.val_coordinates[:, :2]
        )
        self.val_coordinates = np.concatenate(
            [self.val_coordinates, val_extra_feature.reshape(-1, 1)], axis=1
        )

        if self.target_coordinates is None:
            return
        target_extra_feature = query_features(
            tree, values, self.target_coordinates[:, :2]
        )
        self.target_coordinates = np.concatenate(
            [self.target_coordinates, target_extra_feature.reshape(-1, 1)],
            axis=1,
        )

    def _split_train_val(
        self, coordinates: np.ndarray, field: np.ndarray, val_portion: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        log.debug("Splitting train and validation datasets...")
        num_samples = coordinates.shape[0]
        indices = np.arange(num_samples)
        np.random.seed(0)
        np.random.shuffle(indices)
        split_index = int(num_samples * (1 - val_portion))
        train_indices = indices[:split_index]
        val_indices = indices[split_index:]
        return (
            coordinates[train_indices],
            field[train_indices],
            coordinates[val_indices],
            field[val_indices],
        )

    @staticmethod
    def convert_spherical_to_cartesian(
        coordinates: np.ndarray, radius: float = 1.0
    ) -> np.ndarray:
        log.debug("Converting coordinates to cartesian...")
        x = radius * np.cos(coordinates[:, 0]) * np.cos(coordinates[:, 1])
        y = radius * np.cos(coordinates[:, 0]) * np.sin(coordinates[:, 1])
        z = radius * np.sin(coordinates[:, 0])
        return np.stack((x, y, z), axis=1)

    @property
    def train_dataset(self) -> Dataset:
        return torch.utils.data.TensorDataset(
            torch.from_numpy(self.train_coordinates).float(),
            torch.from_numpy(self.train_field).float(),
        )

    @property
    def val_dataset(self) -> Dataset:
        return torch.utils.data.TensorDataset(
            torch.from_numpy(self.val_coordinates).float(),
            torch.from_numpy(self.val_field).float(),
        )

    @property
    def target_dataset(self) -> Dataset:
        if self.target_coordinates is None:
            raise ValueError("Target coordinates are not set.")
        return torch.utils.data.TensorDataset(
            torch.from_numpy(self.target_coordinates).float()
        )
