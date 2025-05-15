from __future__ import annotations

import re
from enum import StrEnum
from typing import Self


class _AxisEnum(StrEnum):
    def __new__(cls, name, regex) -> Self:
        obj = str.__new__(cls, name)
        obj._value_ = name
        obj.code = name
        obj.regex = regex
        return obj


class Axis(_AxisEnum):
    """
    Enum for axis types in a
    [`BaseClimatrixDataset`](/climatrix/api/#climatrix.dataset.base.BaseClimatrixDataset).

    Members
    -------
    - `LATITUDE`: Latitude axis.
    - `LONGITUDE`: Longitude axis.
    - `TIME`: Time axis.
    - `VERTICAL`: Vertical axis.
    - `POINT`: Point axis.

    References
    ----------
    [1] Unidata, MetPy [https://github.com/Unidata/MetPy/blob/main/src/metpy/xarray.py](https://github.com/Unidata/MetPy/blob/main/src/metpy/xarray.py)
    """

    LATITUDE = ("latitude", re.compile(r"^(x?)lat[a-z0-9_]*$"))
    LONGITUDE = ("longitude", re.compile(r"^(x?)lon[a-z0-9_]*$"))
    TIME = ("time", re.compile(r"^(x?)(valid_)?time(s?)([0-9]*)$"))
    VERTICAL = (
        "vertical",
        re.compile(
            r"^(z|lv_|bottom_top|sigma|h(ei)?ght|altitude|depth|"
            r"isobaric|pres|isotherm)"
            r"[a-z_]*[0-9]*$"
        ),
    )
    POINT = ("point", re.compile(r"^(point.*|values|nstation.*)$"))

    @classmethod
    def get(cls, value: str | Axis) -> Axis:
        """
        Get the `Axis` type given by `value`.

        If `value` is an instance of `Axis`,
        return it as is.
        If `value` is a string, return the corresponding
        `Axis`.
        If `value` is neither an instance of `Axis`
        nor a string, raise a ValueError.

        Parameters
        ----------
        value : `str` or `Axis`
            The axis type

        Returns
        -------
        Axis
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
