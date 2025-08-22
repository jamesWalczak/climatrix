from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
import sklearn
import torch
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from torch.utils.data import Dataset

from climatrix.decorators.runtime import log_input

log = logging.getLogger(__name__)


class BaseNNDatasetGenerator(ABC):
    train_coordinates: np.ndarray
    train_field: np.ndarray
    target_coordinates: np.ndarray | None

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
    ) -> None:
        if spatial_points.ndim != 2 or spatial_points.shape[1] != 2:
            raise ValueError(
                "Spatial points must be a 2D array with shape (n_samples, 2)."
            )

        if field.ndim != 1 or field.shape[0] != spatial_points.shape[0]:
            raise ValueError(
                "Field must be a 1D array with the same number of samples as spatial points."
            )

        self.train_coordinates = spatial_points
        self.train_field = field
        self.val_coordinates = self.val_field = None
        if val_portion is not None and val_portion > 0:
            (
                self.train_coordinates,
                self.train_field,
                self.val_coordinates,
                self.val_field,
            ) = self._split_train_val(
                self.train_coordinates, self.train_field, val_portion
            )
        if validation_coordinates is not None or validation_field is not None:
            self.val_coordinates = validation_coordinates
            self.val_field = validation_field

        self.train_coordinates = self.fit_transform_coordinates(spatial_points)
        self.train_field = self.fit_transform_field(field)
        if self.val_coordinates is not None and self.val_field is not None:
            self.val_coordinates = self.transform_coordinates(
                self.val_coordinates
            )
            self.val_field = self.transform_field(self.val_field)

        if target_coordinates is not None and target_coordinates.ndim != 2:
            raise ValueError(
                "Target coordinates must be a 2D array with shape (n_samples, 2)."
            )

        if (
            target_coordinates is not None
            and target_coordinates.shape[1] != spatial_points.shape[1]
        ):
            raise ValueError(
                "Target coordinates must have the same number of dimensions as spatial points."
            )

        self.target_coordinates = None
        if target_coordinates is not None:
            self.target_coordinates = self.transform_coordinates(
                target_coordinates
            )

    @property
    def n_samples(self) -> int:
        """
        Number of samples in the training dataset.

        Returns
        -------
        int
            Number of samples.
        """
        return self.train_coordinates.shape[0]

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

    @property
    def train_dataset(self) -> Dataset:
        return torch.utils.data.TensorDataset(
            torch.from_numpy(self.train_coordinates).float(),
            torch.from_numpy(self.train_field).float(),
        )

    @property
    def val_dataset(self) -> Dataset | None:
        if self.val_coordinates is None or self.val_field is None:
            return None
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

    def fit_transform_coordinates(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Fit and transform the coordinates using the configured transformer.

        Parameters
        ----------
        coordinates : np.ndarray
            Original coordinates.

        Returns
        -------
        np.ndarray
            Transformed coordinates.
        """
        return coordinates

    def transform_coordinates(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Transform the coordinates using the configured transformer.

        Parameters
        ----------
        coordinates : np.ndarray
            Original coordinates.

        Returns
        -------
        np.ndarray
            Transformed coordinates.
        """
        return coordinates

    def untransform_coordinates(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Inverse transform the coordinates using the configured transformer.

        Parameters
        ----------
        coordinates : np.ndarray
            Transformed coordinates.

        Returns
        -------
        np.ndarray
            Coordinates after reverse transformation.
        """
        return coordinates

    def fit_transform_field(self, field: np.ndarray) -> np.ndarray:
        """
        Fit and transform the field using the configured transformer.

        Parameters
        ----------
        field : np.ndarray
            Original field.

        Returns
        -------
        np.ndarray
            Transformed field.
        """
        return field

    def transform_field(self, field: np.ndarray) -> np.ndarray:
        """
        Transform the field using the configured transformer.

        Parameters
        ----------
        field : np.ndarray
            Original field.

        Returns
        -------
        np.ndarray
            Transformed field.
        """
        return field

    def untransform_field(self, field: np.ndarray) -> np.ndarray:
        """
        Inverse transform the field using the configured transformer.

        Parameters
        ----------
        field : np.ndarray
            Transformed field.

        Returns
        -------
        np.ndarray
            Field after reverse transformation.
        """
        return field
