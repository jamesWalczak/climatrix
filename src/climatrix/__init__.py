import importlib
import importlib.resources
from logging import config


def _configure_logger():
    logging_conf_path = importlib.resources.files("climatrix").joinpath(
        "..", "..", "resources", "logging.ini"
    )
    config.fileConfig(logging_conf_path)


_configure_logger()

from .comparison import Comparison
from .dataset.base import BaseClimatrixDataset
