import functools
import inspect
from numbers import Number


def _assert_is_binary_func(func):
    if not callable(func):
        raise TypeError(
            f"Expected a function, but found: {type(func).__name__}"
        )
    arguments = (
        inspect.getfullargspec(func).args
        + inspect.getfullargspec(func).kwonlyargs
    )
    if len(arguments) != 2:
        raise ValueError(
            f"Function '{func.__name__}' must have exactly 2 arguments"
        )


def _assert_first_arg_is_dataset(self_arg):
    from climatrix import BaseClimatrixDataset

    if not isinstance(self_arg, BaseClimatrixDataset):
        raise TypeError("Expected first argument to be BaseClimatrixDataset")


def _get_arguments(func, *args, **kwargs) -> tuple:
    sig = inspect.signature(func)
    args_ = sig.bind(*args, **kwargs)
    args_.apply_defaults()
    arg_1, arg_2 = list(args_.arguments.values())
    return arg_1, arg_2


def _validate_args_types(ds1, ds2):
    from climatrix import BaseClimatrixDataset

    if not isinstance(ds1, BaseClimatrixDataset):
        if not isinstance(ds2, BaseClimatrixDataset):
            raise ValueError(
                "At least one argument must be a BaseClimatrixDataset"
            )
    else:
        if not isinstance(ds2, Number):
            raise ValueError(
                "The other argument must be BaseClimatrixDataset or a number"
            )


def _maybe_validate_domains(ds1, ds2):
    from climatrix import BaseClimatrixDataset

    if isinstance(ds2, BaseClimatrixDataset) and ds1.domain != ds2.domain:
        raise ValueError(
            "Domains for both BaseClimatrixDatasets must be the same."
        )


def cm_arithmetic_binary_operator(func):
    """
    Decorator to facilitates arithmetic binary operators for dataset.

    It enables xarray DataArray arithmetic operations between two
    BaseClimatrixDataset instances.
    """
    _assert_is_binary_func(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        from climatrix import BaseClimatrixDataset

        self_arg, other_arg = _get_arguments(func, *args, **kwargs)
        _assert_first_arg_is_dataset(self_arg)
        _maybe_validate_domains(self_arg, other_arg)
        other_arg = other_arg.transpose(*self_arg.dims)
        if isinstance(other_arg, BaseClimatrixDataset):
            other_arg = other_arg.da.data
        res = getattr(self_arg.da.data, func.__name__)(other_arg)
        return type(self_arg)(
            self_arg.domain.to_xarray(res, name=self_arg.da.name)
        )

    return wrapper
