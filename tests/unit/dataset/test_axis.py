import numpy as np
import pytest

from climatrix.dataset.axis import (
    Axis,
    AxisType,
    Latitude,
    Longitude,
    Point,
    Time,
    Vertical,
)
from climatrix.exceptions import AxisMatchingError


class TestAxis:
    def test_axis_initialization(self):
        axis = Axis(name="lat", values=np.array([1]), is_dimension=True)
        assert axis.name == "lat"
        assert axis.is_dimension is True

    def test_axis_initialization_fail_on_unrecognized_name(self):
        with pytest.raises(
            AxisMatchingError,
            match="No matching axis found for name: unrecognized_axis*",
        ):
            Axis(
                name="unrecognized_axis",
                values=np.array([1]),
                is_dimension=True,
            )

    def test_axis_initialization_empty_on_none(self):
        with pytest.warns(
            match="No values provided. Axis will contain no values"
        ):
            Axis(name="lat", values=None, is_dimension=True)

    def test_axis_matches(self):
        assert Latitude.matches("latitude")
        assert not Latitude.matches("longitude")
        assert Longitude.matches("longitude")
        assert not Longitude.matches("latitude")

    def test_get_all_axes(self):
        all_axes = Axis.get_all_axes()
        assert Latitude in all_axes
        assert Longitude in all_axes
        assert Time in all_axes
        assert Vertical in all_axes
        assert Point in all_axes

    def test_axis_type_get(self):
        assert AxisType.get("latitude") == AxisType.LATITUDE
        assert AxisType.get("longitude") == AxisType.LONGITUDE
        assert AxisType.get("time") == AxisType.TIME
        assert AxisType.get("vertical") == AxisType.VERTICAL
        assert AxisType.get("point") == AxisType.POINT

    def test_Axis_type_get_raise_on_invalid(self):
        with pytest.raises(ValueError):
            AxisType.get("unknown")

        with pytest.raises(TypeError):
            AxisType.get(123)

    def test_latitude_regex(self):
        assert Latitude._regex.match("lat")
        assert Latitude._regex.match("xlat")
        assert not Latitude._regex.match("lon")

    def test_longitude_regex(self):
        assert Longitude._regex.match("lon")
        assert Longitude._regex.match("xlon")
        assert not Longitude._regex.match("lat")

    def test_time_regex(self):
        assert Time._regex.match("time")
        assert Time._regex.match("valid_time")
        assert not Time._regex.match("latitude")

    def test_vertical_regex(self):
        assert Vertical._regex.match("z")
        assert Vertical._regex.match("height")
        assert not Vertical._regex.match("longitude")

    def test_point_regex(self):
        assert Point._regex.match("point")
        assert Point._regex.match("values")
        assert not Point._regex.match("latitude")
