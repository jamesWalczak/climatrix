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


class BaseTransformer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    pass


class IdentityTransformer(BaseTransformer):
    """
    A transformer that does not change the data.
    It is used as a placeholder for the base class.
    """

    def fit(self, X, y=None):
        if y is not None:
            return X, y
        return self

    def transform(self, X):
        return X


class BaseNNDatasetGenerator(ABC):
    lat_lon_transformer: BaseTransformer
    field_transformer: BaseTransformer

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
        self.lat_lon_transformer = self.configure_coordinates_transformer()
        self.field_transformer = self.configure_field_transformer()

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

        self.train_coordinates = self.lat_lon_transformer.fit_transform(
            spatial_points
        )
        self.train_field = self.field_transformer.fit_transform(
            field.reshape(-1, 1)
        ).flatten()
        if self.val_coordinates is not None and self.val_field is not None:
            self.val_coordinates = self.lat_lon_transformer.transform(
                self.val_coordinates
            )
            self.val_field = self.field_transformer.transform(
                self.val_field.reshape(-1, 1)
            ).flatten()

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
            self.target_coordinates = self.lat_lon_transformer.transform(
                target_coordinates
            )

    def configure_coordinates_transformer(self) -> BaseTransformer:
        """
        Configure the transformer for coordinates.
        """
        return IdentityTransformer()

    def configure_field_transformer(self) -> BaseTransformer:
        """
        Configure the transformer for field.
        """
        return IdentityTransformer()

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
