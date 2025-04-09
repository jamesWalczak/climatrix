import functools
import importlib
import logging
from typing import Callable


def raise_if_not_installed(
    *packages, custom_error: Exception | None = None
) -> Callable:
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
                if custom_error:
                    raise custom_error
                raise ImportError(
                    "The following packages are required but not "
                    f"installed: {', '.join(missing)}. "
                    "Please install them using pip or conda before "
                    f"calling '{func.__name__}()'."
                )
            return func(*args, **kwargs)

        return wrapper

    return decorator


def log_input(log: logging.Logger, level: int = logging.INFO):
    """Decorator to facilitte logging the input of a function."""

    def decorator(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            log.log(level, f"Input: {args}, {kwargs}")
            return func(*args, **kwargs)

        return wrapper

    return decorator
