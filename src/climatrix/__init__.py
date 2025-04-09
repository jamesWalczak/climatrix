import importlib
import importlib.resources
from logging import config


def _configure_logger():
    logging_conf_path = importlib.resources.files("climatrix").joinpath(
        "..", "..", "resources", "logging.ini"
    )
    config.fileConfig(logging_conf_path)


_configure_logger()

from .comparison import Comparison as Comparison
from .dataset.base import BaseClimatrixDataset as BaseClimatrixDataset
from .dataset.dense import DenseDataset as DenseDataset
from .dataset.dense import DynamicDenseDataset as DynamicDenseDataset
from .dataset.dense import StaticDenseDataset as StaticDenseDataset
from .dataset.sparse import DynamicSparseDataset as DynamicSparseDataset
from .dataset.sparse import SparseDataset as SparseDataset
from .dataset.sparse import StaticSparseDataset as StaticSparseDataset
