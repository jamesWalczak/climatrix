import numpy as np
import pytest

from climatrix.dataset.axis import Axis
from climatrix.dataset.domain import Domain


def test_ensure_all_numpy_arrays():
    coords = {
        Axis.LATITUDE: [10, 20, 30],
        Axis.LONGITUDE: [10, 20, 30],
    }
    domain = Domain(coords)

    assert isinstance(domain.coords[Axis.LATITUDE], np.ndarray)
    assert isinstance(domain.coords[Axis.LONGITUDE], np.ndarray)


def test_filter_out_single_value_coord():
    coords = {
        Axis.LATITUDE: [10, 20, 30],
        Axis.LONGITUDE: [10, 20, 30],
        Axis.TIME: [12],
    }
    domain = Domain(coords)

    assert Axis.LATITUDE in domain.coords
    assert Axis.LONGITUDE in domain.coords
    assert Axis.TIME not in domain.coords


def test_latitude_property():
    lat_values = np.array([10, 20, 30])
    domain = Domain(
        {Axis.LATITUDE: lat_values, Axis.LONGITUDE: np.array([40, 50, 60])}
    )

    np.testing.assert_array_equal(domain.latitude, lat_values)


def test_latitude_property_raises_error_when_missing():
    domain = Domain({Axis.LONGITUDE: np.array([10, 20, 30])})

    with pytest.raises(ValueError) as excinfo:
        _ = domain.latitude

    assert "Latitude not found" in str(excinfo.value)


def test_longitude_property():
    lon_values = np.array([10, 20, 30])
    domain = Domain(
        {Axis.LATITUDE: np.array([10, 20, 30]), Axis.LONGITUDE: lon_values}
    )

    np.testing.assert_array_equal(domain.longitude, lon_values)


def test_longitude_property_raises_error_when_missing():
    domain = Domain({Axis.LATITUDE: np.array([10, 20, 30])})

    with pytest.raises(ValueError) as excinfo:
        _ = domain.longitude

    assert "Longitude not found" in str(excinfo.value)


def test_size_property_grid_data():
    domain = Domain(
        {
            Axis.LATITUDE: np.array([10, 20, 30]),
            Axis.LONGITUDE: np.array([10, 20, 30]),
        }
    )

    assert domain.size == 9


def test_size_property_point_data():
    point_values = np.array([1, 2, 3, 4, 5])
    domain = Domain(
        {
            Axis.LATITUDE: np.array([10, 20, 30]),
            Axis.LONGITUDE: np.array([10, 20, 30]),
            Axis.POINT: point_values,
        }
    )

    assert domain.size == 5


def test_eq_identical_domains():
    domain1 = Domain(
        {
            Axis.LATITUDE: np.array([10, 20, 30]),
            Axis.LONGITUDE: np.array([10, 20, 30]),
        }
    )

    domain2 = Domain(
        {
            Axis.LATITUDE: np.array([10, 20, 30]),
            Axis.LONGITUDE: np.array([10, 20, 30]),
        }
    )

    assert domain1 == domain2


def test_eq_different_values():
    domain1 = Domain(
        {
            Axis.LATITUDE: np.array([10, 20, 30]),
            Axis.LONGITUDE: np.array([10, 20, 30]),
        }
    )

    domain2 = Domain(
        {
            Axis.LATITUDE: np.array([10, 20, 30]),
            Axis.LONGITUDE: np.array([10, 20, 50]),
        }
    )

    assert domain1 != domain2


def test_eq_different_keys():
    domain1 = Domain(
        {
            Axis.LATITUDE: np.array([10, 20, 30]),
            Axis.LONGITUDE: np.array([10, 20, 30]),
        }
    )

    domain2 = Domain(
        {
            Axis.LATITUDE: np.array([10, 20, 30]),
            Axis.LONGITUDE: np.array([10, 20, 30]),
            Axis.TIME: np.array([12, 13]),
        }
    )

    assert domain1 != domain2


def test_eq_different_types():
    domain = Domain(
        {
            Axis.LATITUDE: np.array([10, 20, 30]),
            Axis.LONGITUDE: np.array([10, 20, 30]),
        }
    )

    not_domain = np.array([10, 20, 30])
    assert domain != not_domain
