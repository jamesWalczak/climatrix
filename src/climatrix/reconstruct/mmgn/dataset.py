from __future__ import annotations

import logging

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from climatrix.reconstruct.nn.dataset import BaseNNDatasetGenerator

log = logging.getLogger(__name__)


class MMGNDatasetGenerator(BaseNNDatasetGenerator):
    coordinate_transformer: MinMaxScaler
    field_transformer: MinMaxScaler

    def fit_transform_coordinates(self, coordinates: np.ndarray) -> np.ndarray:
        self.coordinate_transformer = MinMaxScaler((0, 1))
        return self.coordinate_transformer.fit_transform(coordinates)

    def transform_coordinates(self, coordinates: np.ndarray) -> np.ndarray:
        return self.coordinate_transformer.transform(coordinates)

    def untransform_coordinates(self, coordinates: np.ndarray) -> np.ndarray:
        return self.coordinate_transformer.inverse_transform(coordinates)

    def fit_transform_field(self, field: np.ndarray) -> np.ndarray:
        self.field_transformer = MinMaxScaler((-1, 1))
        return self.field_transformer.fit_transform(
            field.reshape(-1, 1)
        ).ravel()

    def transform_field(self, field: np.ndarray) -> np.ndarray:
        return self.field_transformer.transform(field.reshape(-1, 1)).ravel()

    def untransform_field(self, field: np.ndarray) -> np.ndarray:
        return self.field_transformer.inverse_transform(
            field.reshape(-1, 1)
        ).ravel()
