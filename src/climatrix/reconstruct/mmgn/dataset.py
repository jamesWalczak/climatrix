from __future__ import annotations

import logging

from sklearn.preprocessing import MinMaxScaler

from climatrix.reconstruct.nn.dataset import BaseNNDatasetGenerator

log = logging.getLogger(__name__)


class MMGNDatasetGenerator(BaseNNDatasetGenerator):

    def configure_coordinates_transformer(self):
        return MinMaxScaler((0, 1))

    def configure_field_transformer(self):
        return MinMaxScaler((0, 1))
