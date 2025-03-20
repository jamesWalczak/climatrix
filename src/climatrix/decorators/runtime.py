import functools
import importlib
import inspect
from numbers import Number
from typing import Callable

import numpy as np


def raise_if_not_installed(*packages) -> Callable:
    """
    Decorator to raise an ImportError if a package is not installed.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            missing = []
            for pkg in packages:
                if importlib.util.find_spec(pkg) is None:
                    missing.append(pkg)
            if missing:
                raise ImportError(
                    "The following packages are required but not "
                    f"installed: {', '.join(missing)}. "
                    "Please install them using pip or conda before "
                    f"calling '{func.__name__}()'."
                )
            return func(*args, **kwargs)

        return wrapper

    return decorator
