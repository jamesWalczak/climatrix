import importlib
import importlib.resources
from logging import config

from ._version import __version__


def _configure_logger():
    logging_conf_path = importlib.resources.files("climatrix").joinpath(
        "resources", "logging.ini"
    )
    config.fileConfig(logging_conf_path)


_configure_logger()

from .comparison import Comparison as Comparison
from .dataset.axis import Axis as Axis
from .dataset.axis import AxisType as AxisType
from .dataset.base import BaseClimatrixDataset as BaseClimatrixDataset
from .dataset.domain import Domain as Domain


def seed_all(seed: int):
    """
    Set the seed for all libraries used in the project.
    """
    import random

    import numpy as np

    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    random.seed(seed)
    np.random.seed(seed)
