import functools
import importlib
import inspect
from numbers import Number
from typing import Callable


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


def _arth_binary_operator(func):
    """
    Decorator to facilitates arithmetic binary operators for dataset.

    It enables xarray DataArray arithmetic operations between two
    BaseClimatrixDataset instances.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        from climatrix.dataset.base import BaseClimatrixDataset

        binds = inspect.signature(func).bind(*args, **kwargs)
        binds.apply_defaults()
        arguments = binds.arguments.values()
        if len(arguments) != 2:
            raise ValueError(
                f"Binary operator '{func.__name__}' must have exactly 2 arguments"
            )
        self_dataset, other_dataset = arguments
        if not isinstance(self_dataset, BaseClimatrixDataset):
            raise TypeError(
                f"_arth_binary_operator decorator can be used only with "
                f"BaseClimatrixDataset subclasses, but found: {type(self_dataset).__name__}"
            )
        if isinstance(other_dataset, Number):
            res = getattr(self_dataset.da, func.__name__)(other_dataset)
        elif not isinstance(other_dataset, type(self_dataset)):
            raise TypeError(
                f"Arguments must have the same type, but found: "
                f"{type(self_dataset).__name__} and {type(other_dataset).__name__}"
            )
        else:
            res = getattr(self_dataset.da, func.__name__)(other_dataset.da)
        return type(self_dataset)(res)

    return wrapper
