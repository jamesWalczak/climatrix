from __future__ import annotations

import logging
from collections import namedtuple
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

from climatrix.decorators.runtime import log_input

if TYPE_CHECKING:
    pass

SdfEntry = namedtuple("SdfEntry", ["coordinates", "normals", "sdf"])

log = logging.getLogger(__name__)


class SiNETDatasetGenerator:

    @log_input(log, level=logging.DEBUG)
    def __init__(
        self,
        train_latitudes: np.ndarray,
        train_longitudes: np.ndarray,
        train_field: np.ndarray,
        *,
        degree: bool = True,
        radius: float = 1.0,
    ) -> None:
        """
        Initialize a SiNET dataset generator.

        Parameters
        ----------
        train_latitudes : np.ndarray
            The latitudes of the training points.
        train_longitudes : np.ndarray
            The longitudes of the training points.
        train_field : np.ndarray
            The field values at the training points.
        target_latitudes : np.ndarray
            The latitudes of the target points.
        target_longitudes : np.ndarray
            The longitudes of the target points.
        degree : bool, optional
            Whether the input latitudes and longitudes are in degrees.
            Defaults to True.
        radius : float, optional
            The radius of the sphere. Defaults to 1.0.
        """
        if degree:
            log.debug("Converting degrees to radians...")
            train_latitudes = np.deg2rad(train_latitudes)
            train_longitudes = np.deg2rad(train_longitudes)
        self.radius = radius
        self.train_coordinates = np.stack(
            (train_latitudes, train_longitudes), axis=1
        )
        # self.train_coordinates = (
        #     SiNETDatasetGenerator.convert_spherical_to_cartesian(
        #         np.stack((train_latitudes, train_longitudes), axis=1),
        #         self.radius,
        #     )
        # )
        self.field_transformer = MinMaxScaler((-1, 1))
        # self.train_field = train_field.reshape(-1, 1)
        self.train_field = self.field_transformer.fit_transform(
            train_field.reshape(-1, 1)
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

    def set_target_coordinates(
        self,
        target_latitudes: np.ndarray,
        target_longitudes: np.ndarray,
        degree: bool = True,
    ) -> None:
        if degree:
            target_latitudes = np.deg2rad(target_latitudes)
            target_longitudes = np.deg2rad(target_longitudes)
        self.target_coordinates = np.stack(
            (target_latitudes, target_longitudes), axis=1
        )
        # self.target_coordinates = (
        #     SiNETDatasetGenerator.convert_spherical_to_cartesian(
        #         np.stack((target_latitudes, target_longitudes), axis=1),
        #         self.radius,
        #     )
        # )

    @property
    def train_dataset(self) -> Dataset:
        return torch.utils.data.TensorDataset(
            torch.from_numpy(self.train_coordinates).float(),
            torch.from_numpy(self.train_field).float(),
        )

    @property
    def target_dataset(self) -> Dataset:
        if self.target_coordinates is None:
            raise ValueError("Target coordinates are not set.")
        return torch.utils.data.TensorDataset(
            torch.from_numpy(self.target_coordinates).float()
        )
