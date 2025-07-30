"""
Hyperparameter descriptor for reconstruction methods.
"""

from __future__ import annotations

import numbers
from typing import Any


class Hyperparameter:
    """
    Descriptor class for hyperparameters with validation logic.

    This descriptor validates hyperparameters by type,
    bounds (for numeric types), or valid values (for categorical types)
    when they are accessed or assigned.

    Parameters
    ----------
    param_type : type
        The expected type for the hyperparameter (int, float, str, etc.).
    bounds : tuple, optional
        For numeric types, a tuple of (min_value, max_value) bounds.
    values : list, optional
        For categorical types, a list of valid values.
    default : Any, optional
        Default value for the hyperparameter.

    Examples
    --------
    >>> class SomeReconstructor:
    ...     power = Hyperparameter(float, bounds=(0.5, 5.0), default=2.0)
    ...     k = Hyperparameter(int, bounds=(1, 20), default=5)
    ...     mode = Hyperparameter(str, values=['fast', 'slow'], default='fast')
    """

    def __init__(
        self,
        param_type: type,
        *,
        bounds: tuple[int | float, int | float] | None = None,
        values: list[int | float | str] | None = None,
        default: Any = None,
    ):
        self.param_type = param_type
        self.bounds = bounds
        self.values = values
        self.default = default
        self.name = None
        self.private_name = None

        if bounds is not None and values is not None:
            raise ValueError("Cannot specify both bounds and values")

        if bounds is not None:
            if not (
                isinstance(param_type, type)
                and issubclass(param_type, numbers.Number)
            ):
                raise ValueError(
                    "Bounds can only be specified for numeric types"
                )
            if len(bounds) != 2:
                raise ValueError(
                    "Bounds must be a tuple of (min_value, max_value)"
                )
            if bounds[0] >= bounds[1]:
                raise ValueError("Lower bound must be less than upper bound")

    def __set_name__(self, owner, name):
        """Called when the descriptor is assigned to a class attribute."""
        self.name = name
        self.private_name = f"_{name}"

    def __get__(self, instance, owner):
        """Get the hyperparameter value."""
        if instance is None:
            return self

        return getattr(instance, self.private_name, self.default)

    def __set__(self, instance, value):
        """Set the hyperparameter value with validation."""
        validated_value = self._validate_and_cast(value)
        setattr(instance, self.private_name, validated_value)

    def _validate_and_cast(self, value):
        """Validate and cast the value according to the hyperparameter specification."""
        if value is None:
            return self.default

        try:
            casted_value = self.param_type(value)
        except (ValueError, TypeError) as e:
            raise TypeError(
                f"Cannot convert {value!r} to {self.param_type.__name__} for parameter '{self.name}'"
            ) from e

        if self.bounds is not None and isinstance(
            casted_value, numbers.Number
        ):
            min_val, max_val = self.bounds
            if not (min_val <= casted_value <= max_val):
                raise ValueError(
                    f"Parameter '{self.name}' value {casted_value} is outside bounds [{min_val}, {max_val}]"
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
        spec = {"type": self.param_type}

        if self.bounds is not None:
            spec["bounds"] = self.bounds

        if self.values is not None:
            spec["values"] = self.values

        if self.default is not None:
            spec["default"] = self.default

        return spec
