from __future__ import annotations

import logging
import re
import warnings
from enum import StrEnum
from typing import ClassVar, final

import numpy as np

from climatrix.exceptions import AxisMatchingError

log = logging.getLogger(__name__)


class AxisType(StrEnum):
    """
    Enum for axis types.

    Attributes
    ----------
    LATITUDE : str
        Latitude axis type.
    LONGITUDE : str
        Longitude axis type.
    TIME : str
        Time axis type.
    VERTICAL : str
        Vertical axis type.
    POINT : str
        Point axis type.
    """

    LATITUDE = "latitude"
    LONGITUDE = "longitude"
    TIME = "time"
    VERTICAL = "vertical"
    POINT = "point"

    @classmethod
    def get(cls, value: str | Axis) -> Axis:
        """
        Get the `AxisType` type given by `value`.

        If `value` is an instance of `AxisType`,
        return it as is.
        If `value` is a string, return the corresponding
        `AxisType`.
        If `value` is neither an instance of `AxisType`
        nor a string, raise a ValueError.

        Parameters
        ----------
        value : str or AxisType
            The axis type

        Returns
        -------
        AxisType
            The axis type.

        Raises
        ------
        ValueError
            If `value` is not a valid axis type.
        """
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise TypeError(
                f"Invalid axis type: {value!r}. "
                "Expected a string or an instance of Axis."
            )
        try:
            return cls[value.upper()]
        except KeyError:
            raise ValueError(f"Unknown axis type: {value}")


class Axis:
    """
    Base class for axis types.

    Attributes
    ----------
    type : ClassVar[AxisType]
        The type of the axis.
    dtype : ClassVar[np.dtype]
        The data type of the axis values.
    is_dimension : bool
        Whether the axis is a dimension or not.
    name : str
        The name of the axis.
    values : np.ndarray
        The values of the axis.

    Parameters
    ----------
    name : str
        The name of the axis.
    values : np.ndarray
        The values of the axis.
    is_dimension : bool, optional
        Whether the axis is a dimension or not (default is True).

    Examples
    --------
    Axis is a factory class for all axis types. To create an axis (by
    matching the name), use:
    >>> axis = Axis(name="latitude", values=np.array([1, 2, 3]))

    To create a `Latitude` axis explicitly, use:
    >>> axis = Latitude(name="latitude", values=np.array([1, 2, 3]))
    >>> axis = Latitude(
    ... name="latitude",
    ... values=np.array([1, 2, 3]),
    ... is_dimension=True)

    Notes
    -----
    - The `Axis` class is a factory class for all axis types.
    - If the given axis has "unusual" name, you need to create it
        explicitly using the corresponding class (e.g. `Latitude`).
    """

    _regex: ClassVar[re.Pattern]
    dtype: ClassVar[np.dtype] = np.dtype("float32")
    type: ClassVar[AxisType]
    is_dimension: bool
    name: str
    values: np.ndarray

    def __new__(cls, name: str, values: np.ndarray, is_dimension: bool = True):
        """
        Create a new instance of the Axis class.

        Parameters
        ----------
        name : str
            The name of the axis.
        values : np.ndarray
            The values of the axis.
        is_dimension : bool, optional
            Whether the axis is a dimension or not (default is True).
        """
        for axis in cls.get_all_axes():
            if axis.matches(name):
                return super().__new__(axis)
        log.error(
            "No matching axis found for name: %s. "
            "Create explicitly one of the predefined axes "
            "(e.g. `Latitude(name='custom_name', values, True)`)",
            name,
        )
        raise AxisMatchingError(
            f"No matching axis found for name: {name}. "
            "Create explicitly one of the predefined axes "
            "(e.g. `Latitude(name='custom_name', values, True)`)",
        )

    def __init__(
        self,
        name: str,
        values: np.ndarray,
        is_dimension: bool = True,
    ):
        self.name = name
        self.is_dimension = is_dimension
        if values is None:
            log.warning("No values provided. Axis will contain no values")
            warnings.warn("No values provided. Axis will contain no values")
            values = []
        values = np.asarray(values, dtype=self.dtype)
        self.values = values

    def __eq__(self, other: object) -> bool:
        """
        Check if two axes are equal.

        Parameters
        ----------
        other : object
            The other object to compare with.

        Returns
        -------
        bool
            True if the axes are equal, False otherwise.
        """
        if not isinstance(other, Axis):
            return False
        return (
            self.name == other.name
            and len(self.values) == len(other.values)
            and np.allclose(self.values, other.values, equal_nan=True)
            and self.is_dimension == other.is_dimension
        )

    @classmethod
    def matches(cls, name: str) -> bool:
        """
        Check if the axis matches the given name.

        Parameters
        ----------
        name : str
            The name to check.

        Returns
        -------
        bool
            True if the axis matches the name, False otherwise.
        """
        return bool(cls._regex.match(name))

    @property
    def size(self) -> int:
        """
        Get the size of the axis.

        Returns
        -------
        int
            The size of the axis.
        """
        return len(self.values) if self.values is not None else 0

    @final
    @classmethod
    def get_all_axes(cls) -> list[type[Axis]]:
        """
        Get all axis classes.

        Returns
        -------
        list[Type[Axis]]
            A list of all axis classes.
        """
        return cls.__subclasses__()


class Latitude(Axis):
    """
    Latitude axis.

    Attributes
    ----------
    name : str
        The name of the latitude axis.
    is_dimension : bool
        Whether the axis is a dimension or not.
    """

    _regex = re.compile(r"^(x?)lat[a-z0-9_]*$")
    type = AxisType.LATITUDE
    dtype: ClassVar[np.dtype] = np.dtype("float32")


class Longitude(Axis):
    """
    Longitude axis.

    Attributes
    ----------
    name : str
        The name of the longitude axis.
    is_dimension : bool
        Whether the axis is a dimension or not.
    """

    _regex = re.compile(r"^(x?)lon[a-z0-9_]*$")
    type = AxisType.LONGITUDE
    dtype: ClassVar[np.dtype] = np.dtype("float32")


class Time(Axis):
    """
    Time axis.

    Attributes
    ----------
    name : str
        The name of the time axis.
    is_dimension : bool
        Whether the axis is a dimension or not.
    """

    _regex = re.compile(r"^(x?)(valid_)?time(s?)([0-9]*)$")
    type = AxisType.TIME
    dtype: ClassVar[np.dtype] = np.dtype("datetime64[ns]")

    def __eq__(self, other: object) -> bool:
        """
        Check if two axes are equal.

        Parameters
        ----------
        other : object
            The other object to compare with.

        Returns
        -------
        bool
            True if the axes are equal, False otherwise.
        """
        if not isinstance(other, Axis):
            return False
        return (
            self.name == other.name
            and np.array_equal(self.values, other.values)
            and self.is_dimension == other.is_dimension
        )


class Vertical(Axis):
    """
    Vertical axis.

    Attributes
    ----------
    name : str
        The name of the vertical axis.
    is_dimension : bool
        Whether the axis is a dimension or not.
    """

    _regex = re.compile(
        r"^(z|lv_|bottom_top|sigma|h(ei)?ght|altitude|depth|"
        r"isobaric|pres|isotherm)[a-z_]*[0-9]*$"
    )
    type = AxisType.VERTICAL
    dtype: ClassVar[np.dtype] = np.dtype("float32")


class Point(Axis):
    """
    Point axis.

    Attributes
    ----------
    name : str
        The name of the point axis.
    is_dimension : bool
        Whether the axis is a dimension or not.
    """

    _regex = re.compile(r"^(point.*|values|nstation.*)$")
    type = AxisType.POINT
    dtype: ClassVar[np.dtype] = np.dtype("int")
