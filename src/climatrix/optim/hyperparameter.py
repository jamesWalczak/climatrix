"""
Hyperparameter descriptor for reconstruction methods.
"""

from __future__ import annotations

import numbers
from typing import Any, Generic, Self, Type, TypeVar, overload

T = TypeVar("T")


class Hyperparameter(Generic[T]):
    """
    Descriptor class for hyperparameters with validation logic.

    This descriptor validates hyperparameters by type,
    bounds (for numeric types), or valid values (for categorical types)
    when they are accessed or assigned.

    Type Parameters
    ---------------
    T : type
        The type of the hyperparameter (int, float, str, etc.).

    Parameters
    ----------
    bounds : tuple, optional
        For numeric types, a tuple of (min_value, max_value) bounds.
    values : list, optional
        For categorical types, a list of valid values.
    default : T, optional
        Default value for the hyperparameter.

    Examples
    --------
    >>> class SomeReconstructor:
    ...     power: Hyperparameter[float] = Hyperparameter(bounds=(0.5, 5.0), default=2.0)
    ...     k: Hyperparameter[int] = Hyperparameter(bounds=(1, 20), default=5)
    ...     mode: Hyperparameter[str] = Hyperparameter(values=['fast', 'slow'], default='fast')
    """

    def __init__(
        self,
        *,
        default: T,
        bounds: tuple[int | float | None, int | float | None] | None = None,
        values: list[T] | None = None,
    ):
        self.param_type: type[T] | None = None
        self.bounds = bounds
        self.default_bounds = bounds
        self.values = values
        self.default = default
        self.name: str | None = None
        self.private_name: str | None = None

        if bounds is not None and values is not None:
            raise ValueError("Cannot specify both bounds and values")

        if bounds is not None:
            if len(bounds) != 2:
                raise ValueError(
                    "Bounds must be a tuple of (min_value, max_value)"
                )
            if bounds[0] is not None and bounds[1] is not None:
                if bounds[0] >= bounds[1]:
                    raise ValueError(
                        "Lower bound must be less than upper bound"
                    )

    def __class_getitem__(cls, item):
        class _HyperparameterType(cls):
            param_type = item

            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.param_type = item

                if self.bounds is not None:
                    if not (
                        isinstance(item, type)
                        and issubclass(item, numbers.Number)
                    ):
                        raise ValueError(
                            "Bounds can only be specified for numeric types"
                        )

        return _HyperparameterType

    def __set_name__(self, owner, name):
        """Called when the descriptor is assigned to a class attribute."""
        self.name = name
        self.private_name = f"_{name}"

    @overload
    def __get__(self, instance: None, owner: type) -> Self: ...  # noqa: E704

    @overload
    def __get__(self, instance: Any, owner: type) -> T: ...  # noqa: E704

    def __get__(self, instance: Any | None, owner: type) -> Self | T:
        """Get the hyperparameter value."""
        if instance is None:
            return self

        if self.private_name is None:
            raise RuntimeError("Hyperparameter not properly initialized")

        return getattr(instance, self.private_name, self.default)

    def __set__(self, instance, value):
        """Set the hyperparameter value with validation."""
        if self.private_name is None:
            raise RuntimeError("Hyperparameter not properly initialized")

        validated_value = self._validate_and_cast(value)
        setattr(instance, self.private_name, validated_value)

    def _validate_and_cast(self, value):
        """Validate and cast the value according to the hyperparameter specification."""
        if value is None:
            return self.default

        if self.param_type is None:
            raise RuntimeError(
                "Parameter type not set. Use Hyperparameter[Type] syntax."
            )

        try:
            casted_value = self.param_type(value)  # type: ignore
        except (ValueError, TypeError) as e:
            raise TypeError(
                f"Cannot convert {value!r} to {self.param_type.__name__} for parameter '{self.name}'"
            ) from e

        if self.bounds is not None and isinstance(
            casted_value, numbers.Number
        ):
            min_val, max_val = self.bounds
            if min_val is not None and casted_value < min_val:  # type: ignore
                raise ValueError(
                    f"Parameter '{self.name}' value {casted_value} is below the minimum bound {min_val}"
                )
            if max_val is not None and casted_value > max_val:  # type: ignore
                raise ValueError(
                    f"Parameter '{self.name}' value {casted_value} is above the maximum bound {max_val}"
                )

        if self.values is not None:
            if casted_value not in self.values:
                raise ValueError(
                    f"Parameter '{self.name}' value {casted_value!r} not in valid values {self.values}"
                )

        return casted_value

    def get_spec(self) -> dict[str, Any]:
        """
        Get the hyperparameter specification as a dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary containing the hyperparameter specification with keys:
            - 'type': The parameter type
            - 'bounds': Bounds tuple (if specified)
            - 'values': List of valid values (if specified)
            - 'default': Default value (if specified)
        """
        spec: dict[str, Any] = {"type": self.param_type}

        if self.bounds is not None:
            spec["bounds"] = self.bounds

        if self.values is not None:
            spec["values"] = self.values

        if self.default is not None:
            spec["default"] = self.default

        return spec

    def restore_default_bounds(self) -> None:
        """Restore the bounds to their original default values."""
        self.bounds = self.default_bounds
