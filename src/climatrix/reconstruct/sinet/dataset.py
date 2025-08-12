from __future__ import annotations

import importlib.resources
import logging
from collections import namedtuple
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

from climatrix.decorators.runtime import log_input
from climatrix.reconstruct.nn.dataset import BaseNNDatasetGenerator

SdfEntry = namedtuple("SdfEntry", ["coordinates", "normals", "sdf"])

log = logging.getLogger(__name__)

_ELEVATION_DATASET_PATH: Path = importlib.resources.files(
    "climatrix.reconstruct.sinet"
).joinpath("resources", "lat_lon_elevation.npy")


def load_elevation_dataset() -> np.ndarray:
    """
    Load the elevation dataset (download if needed).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple containing the coordinates and elevation data as numpy arrays.
        The first array contains the coordinates (latitude, longitude),
        and the second array contains the elevation values.
    """
    log.debug("Loading elevation dataset...")
    try:
        data = np.load(_ELEVATION_DATASET_PATH)
    except FileNotFoundError:
        log.info("Elevation dataset not found, downloading...")
        # TODO: download to be implemented
        raise NotImplementedError
        log.info("Elevation dataset downloaded successfully.")
        data = np.load(_ELEVATION_DATASET_PATH)
    return data[:, :-1], MinMaxScaler((-1, 1)).fit_transform(
        data[:, -1].reshape(-1, 1)
    )


def query_features(
    tree: cKDTree, values: np.ndarray, query_points: np.ndarray
):
    log.debug("Querying nearest neighbours...")
    distances, indices = tree.query(query_points, k=1)
    if np.any(distances > 0.1):
        log.warning(
            "Some coordinates are too far from the known data points. The maximum distance is %f.",
            distances.max(),
        )
    return values[indices]


class SiNETDatasetGenerator(BaseNNDatasetGenerator):
    radius: float

    @log_input(log, level=logging.DEBUG)
    def __init__(
        self,
        spatial_points: np.ndarray,
        field: np.ndarray,
        *,
        target_coordinates: np.ndarray | None = None,
        val_portion: float | None = None,
        validation_coordinates: np.ndarray | None = None,
        validation_field: np.ndarray | None = None,
        degree: bool = True,
        radius: float = 1.0,
        use_elevation: bool = False,
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
        val_portion : float | None, optional
            Portion of the training data to use for validation. Defaults to 0.2.
        validation_coordinates : np.ndarray | None, optional
            Array on shape Nx2 with latitudes and longitudes of validation points.
            Cannot be used together with `val_portion`.
            Must be provided if `validation_field` is given.
        validation_field : np.ndarray | None, optional
            Array of shape (N,) with field values at validation points.
            Cannot be used together with `val_portion`.
            Must be provided if `validation_coordinates` is given.
        use_elevation: bool, optional
            Whether to use elevation data. Defaults to False.
        """
        super().__init__(
            spatial_points=spatial_points,
            field=field,
            target_coordinates=target_coordinates,
            val_portion=val_portion,
            validation_coordinates=validation_coordinates,
            validation_field=validation_field,
        )
        if degree:
            log.debug("Converting degrees to radians...")
            spatial_points = np.deg2rad(spatial_points)
            if target_coordinates is not None:
                self.target_coordinates = np.deg2rad(target_coordinates)

        ckdtree = None
        self.radius = radius
        if use_elevation:
            coords, self.elevation = load_elevation_dataset()
            ckdtree = cKDTree(np.deg2rad(coords))
            self._extend_input_features(ckdtree, self.elevation)

    def configure_field_transformer(self):
        return MinMaxScaler((0, 1))

    def _extend_input_features(
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

    @staticmethod
    def convert_spherical_to_cartesian(
        coordinates: np.ndarray, radius: float = 1.0
    ) -> np.ndarray:
        log.debug("Converting coordinates to cartesian...")
        x = radius * np.cos(coordinates[:, 0]) * np.cos(coordinates[:, 1])
        y = radius * np.cos(coordinates[:, 0]) * np.sin(coordinates[:, 1])
        z = radius * np.sin(coordinates[:, 0])
        return np.stack((x, y, z), axis=1)
