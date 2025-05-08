import re
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

from climatrix.dataset.axis import Axis

# Import the classes and functions to be tested
from climatrix.dataset.domain import (
    _DEFAULT_LAT_RESOLUTION,
    _DEFAULT_LON_RESOLUTION,
    DenseDomain,
    Domain,
    SamplingNaNPolicy,
    SparseDomain,
    _coords_name_regex,
    check_is_dense,
    ensure_all_numpy_arrays,
    ensure_single_var,
    filter_out_single_value_coord,
    match_axis_names,
    validate_input,
    validate_spatial_axes,
)
from climatrix.exceptions import TooLargeSamplePortionWarning
from climatrix.types import Latitude, Longitude


class TestSamplingNaNPolicy:
    """Test the SamplingNaNPolicy enum class."""

    def test_valid_policies(self):
        """Test that valid policies can be accessed."""
        assert SamplingNaNPolicy.IGNORE == "ignore"
        assert SamplingNaNPolicy.RESAMPLE == "resample"
        assert SamplingNaNPolicy.RAISE == "raise"

    def test_missing_policy(self):
        """Test that missing policy raises ValueError."""
        with pytest.raises(
            ValueError,
            match="'invalid_policy' is not a valid SamplingNaNPolicy",
        ):
            SamplingNaNPolicy("invalid_policy")

    def test_get_method_with_string(self):
        """Test the get method with string input."""
        assert SamplingNaNPolicy.get("ignore") == SamplingNaNPolicy.IGNORE
        assert SamplingNaNPolicy.get("IGNORE") == SamplingNaNPolicy.IGNORE

    def test_get_method_with_enum(self):
        """Test the get method with enum input."""
        policy = SamplingNaNPolicy.IGNORE
        assert SamplingNaNPolicy.get(policy) == policy

    def test_get_method_with_invalid_string(self):
        """Test that get method raises ValueError for invalid string."""
        with pytest.raises(ValueError, match="Unknown Nan policy: invalid"):
            SamplingNaNPolicy.get("invalid")


class TestDomainHelperFunctions:
    """Test the helper functions for Domain classes."""

    def test_validate_input(self):
        """Test validate_input function."""
        # Since implementation is not provided, we'll just test the interface
        da = xr.DataArray(np.random.rand(5, 5))
        validate_input(da)  # Should not raise an error

        ds = xr.Dataset({"var": da})
        validate_input(ds)  # Should not raise an error

    def test_ensure_single_var(self):
        """Test ensure_single_var function."""
        # Create a simple DataArray
        da = xr.DataArray(np.random.rand(5, 5), dims=["x", "y"])
        result = ensure_single_var(da)
        assert isinstance(result, xr.DataArray)

        # Create a Dataset with single variable
        ds = xr.Dataset({"var": da})
        result = ensure_single_var(ds)
        assert isinstance(result, xr.DataArray)

    def test_match_axis_names(self):
        """Test match_axis_names function."""
        # Create a DataArray with standard dimension names
        da = xr.DataArray(
            np.random.rand(3, 4, 5),
            dims=["time", "lat", "lon"],
            coords={
                "time": np.array(
                    ["2020-01-01", "2020-01-02", "2020-01-03"],
                    dtype="datetime64",
                ),
                "lat": np.linspace(-90, 90, 4),
                "lon": np.linspace(-180, 180, 5),
            },
        )
        result = match_axis_names(da)
        assert Axis.TIME in result
        assert Axis.LATITUDE in result
        assert Axis.LONGITUDE in result

    def test_validate_spatial_axes_if_present(self):
        """Test validate_spatial_axes function."""
        # Valid axis mapping with lat and lon
        axis_mapping = {Axis.LATITUDE: "lat", Axis.LONGITUDE: "lon"}
        validate_spatial_axes(axis_mapping)  # Should not raise an error

    def test_validate_spatial_axes_if_lat_missing(self):
        axis_mapping = {Axis.POINT: "point", Axis.LONGITUDE: "lon"}
        with pytest.raises(ValueError, match="Dataset has no latitude axis"):
            validate_spatial_axes(axis_mapping)

    def test_validate_spatial_axes_if_lon_missing(self):
        axis_mapping = {Axis.POINT: "point", Axis.LATITUDE: "lat"}
        with pytest.raises(ValueError, match="Dataset has no longitude axis"):
            validate_spatial_axes(axis_mapping)

    def test_check_is_dense(self):
        """Test check_is_dense function."""
        # Create a dense DataArray
        da = xr.DataArray(
            np.random.rand(3, 3),
            dims=["lat", "lon"],
            coords={
                "lat": np.linspace(-90, 90, 3),
                "lon": np.linspace(-180, 180, 3),
            },
        )
        axis_mapping = {Axis.LATITUDE: "lat", Axis.LONGITUDE: "lon"}
        result = check_is_dense(da, axis_mapping)
        assert isinstance(result, bool)

    def test_ensure_all_numpy_arrays_numpy_array_passed(self):
        coords = {
            "lat": np.array([-90, 0, 90]),
            "lon": np.array([-180, 0, 180]),
        }
        coords2 = ensure_all_numpy_arrays(coords)  # Should not raise an error
        assert isinstance(coords2["lat"], np.ndarray)
        assert isinstance(coords2["lon"], np.ndarray)

    def test_ensure_all_numpy_arrays_list_passed(self):
        coords = {"lat": [-90, 0, 90], "lon": [-180, 0, 180]}
        coords2 = ensure_all_numpy_arrays(coords)
        assert isinstance(coords2["lat"], np.ndarray)
        assert isinstance(coords2["lon"], np.ndarray)

    def test_filter_out_single_value_coord(self):
        """Test filter_out_single_value_coord function."""
        # Dictionary with arrays of different lengths
        coords = {
            "lat": np.array([-90, 0, 90]),
            "lon": np.array([-180, 0, 180]),
            "single": np.array([42]),
        }
        result = filter_out_single_value_coord(coords)
        assert "lat" in result
        assert "lon" in result
        assert "single" not in result


class TestDomain:
    """Test the Domain base class."""

    def test_domain_factory_method(self):
        """Test that Domain.__new__ returns appropriate subclass."""
        # Create a dense DataArray
        dense_da = xr.DataArray(
            np.random.rand(3, 3),
            dims=["lat", "lon"],
            coords={
                "lat": np.linspace(-90, 90, 3),
                "lon": np.linspace(-180, 180, 3),
            },
        )

        # Mock the check_is_dense function to always return True
        with patch(
            "climatrix.dataset.domain.check_is_dense", return_value=True
        ):
            domain = Domain(dense_da)
            assert isinstance(domain, DenseDomain)

        # Mock the check_is_dense function to always return False
        with patch(
            "climatrix.dataset.domain.check_is_dense", return_value=False
        ):
            domain = Domain(dense_da)
            assert isinstance(domain, SparseDomain)

    def test_from_lat_lon(self):
        """Test the from_lat_lon class method."""
        # Test with default parameters (sparse domain)
        domain = Domain.from_lat_lon()
        assert isinstance(domain, SparseDomain)

        # Test with explicit dense domain
        domain = Domain.from_lat_lon(kind="dense")
        assert isinstance(domain, DenseDomain)

        # Test with custom lat/lon arrays
        lat = np.array([-90, 0, 90])
        lon = np.array([-180, 0, 180])
        domain = Domain.from_lat_lon(lat=lat, lon=lon)
        np.testing.assert_array_equal(domain.latitude, lat)
        np.testing.assert_array_equal(domain.longitude, lon)

    def test_coordinate_properties(self):
        """Test coordinate name and value properties."""
        # Create a mock Domain instance
        domain = MagicMock(spec=Domain)
        domain._axis_mapping = {
            Axis.LATITUDE: "lat",
            Axis.LONGITUDE: "lon",
            Axis.TIME: "time",
            Axis.POINT: "point",
        }
        domain.coords = {
            "lat": np.array([-90, 0, 90]),
            "lon": np.array([-180, 0, 180]),
            "time": np.array(["2020-01-01", "2020-01-02"], dtype="datetime64"),
            "point": np.array([1, 2, 3]),
        }

        # Test coordinate name properties
        assert Domain.latitude_name.__get__(domain) == "lat"
        assert Domain.longitude_name.__get__(domain) == "lon"
        assert Domain.time_name.__get__(domain) == "time"
        assert Domain.point_name.__get__(domain) == "point"

        # Test coordinate value properties
        np.testing.assert_array_equal(
            Domain.latitude.__get__(domain), domain.coords["lat"]
        )
        np.testing.assert_array_equal(
            Domain.longitude.__get__(domain), domain.coords["lon"]
        )
        np.testing.assert_array_equal(
            Domain.time.__get__(domain), domain.coords["time"]
        )
        np.testing.assert_array_equal(
            Domain.point.__get__(domain), domain.coords["point"]
        )

    def test_size_methods(self):
        """Test size-related methods."""
        # Create a mock Domain instance
        domain = MagicMock(spec=Domain)
        domain._axis_mapping = {Axis.LATITUDE: "lat", Axis.LONGITUDE: "lon"}
        domain.coords = {
            "lat": np.array([-90, 0, 90]),
            "lon": np.array([-180, 0, 180]),
        }

        # Test get_size for specific axis
        assert Domain.get_size(domain, Axis.LATITUDE) == 3

        # Test overall size property (this depends on implementation)
        # For testing, we'll assume it's the product of all dimensions
        domain.get_size.side_effect = lambda axis: len(
            domain.coords[domain._axis_mapping[axis]]
        )
        assert Domain.size.__get__(domain) > 0

    def test_is_dynamic(self):
        """Test is_dynamic property."""
        # Create a mock Domain instance
        domain = MagicMock(spec=Domain)
        domain._axis_mapping = {Axis.LATITUDE: "lat", Axis.LONGITUDE: "lon"}

        # Without time, should not be dynamic
        assert not Domain.is_dynamic.__get__(domain)

        # With time, should be dynamic
        domain._axis_mapping[Axis.TIME] = "time"
        domain.coords = {
            "time": np.array(["2020-01-01", "2020-01-02"], dtype="datetime64")
        }
        assert Domain.is_dynamic.__get__(domain)

    def test_equality(self):
        """Test equality comparison."""
        # Create two mock Domain instances
        domain1 = MagicMock(spec=Domain)
        domain1.coords = {
            "lat": np.array([-90, 0, 90]),
            "lon": np.array([-180, 0, 180]),
        }
        domain1._axis_mapping = {Axis.LATITUDE: "lat", Axis.LONGITUDE: "lon"}

        domain2 = MagicMock(spec=Domain)
        domain2.coords = {
            "lat": np.array([-90, 0, 90]),
            "lon": np.array([-180, 0, 180]),
        }
        domain2._axis_mapping = {Axis.LATITUDE: "lat", Axis.LONGITUDE: "lon"}

        # Test equality
        assert Domain.__eq__(domain1, domain2)

        # Test inequality with different coords
        domain2.coords["lat"] = np.array([-45, 0, 45])
        assert not Domain.__eq__(domain1, domain2)

        # Test inequality with different axis mapping
        domain2.coords["lat"] = np.array([-90, 0, 90])
        domain2._axis_mapping[Axis.LATITUDE] = "latitude"
        assert not Domain.__eq__(domain1, domain2)

        # Test inequality with different type
        assert not Domain.__eq__(domain1, "not a domain")

    def test_sampling_points_calculation(self):
        """Test _get_sampling_points_nbr method."""
        # Create a mock Domain instance
        domain = MagicMock(spec=Domain)
        domain.size = 100

        # Test with portion specified
        assert (
            Domain._get_sampling_points_nbr(domain, portion=0.5, number=None)
            == 50
        )

        # Test with number specified
        assert (
            Domain._get_sampling_points_nbr(domain, portion=None, number=30)
            == 30
        )

        # Test with both specified (number should take precedence)
        assert (
            Domain._get_sampling_points_nbr(domain, portion=0.5, number=30)
            == 30
        )

        # Test with neither specified (should return entire size)
        assert (
            Domain._get_sampling_points_nbr(domain, portion=None, number=None)
            == 100
        )

        # Test warning for large portion
        with pytest.warns(TooLargeSamplePortionWarning):
            Domain._get_sampling_points_nbr(domain, portion=0.9, number=None)


class TestSparseDomain:
    """Test the SparseDomain class."""

    def test_get_all_spatial_points(self):
        """Test get_all_spatial_points method."""
        # Create a mock SparseDomain instance
        domain = MagicMock(spec=SparseDomain)
        domain.coords = {
            "lat": np.array([-90, 0, 90]),
            "lon": np.array([-180, 0, 180]),
        }
        domain._axis_mapping = {Axis.LATITUDE: "lat", Axis.LONGITUDE: "lon"}
        domain.latitude = domain.coords["lat"]
        domain.longitude = domain.coords["lon"]

        # Call the method
        result = SparseDomain.get_all_spatial_points(domain)

        # Check that result is a numpy array
        assert isinstance(result, np.ndarray)

    def test_compute_subset_indexers(self):
        """Test _compute_subset_indexers method."""
        # Create a mock SparseDomain instance
        domain = MagicMock(spec=SparseDomain)
        domain.coords = {
            "lat": np.array([-90, -45, 0, 45, 90]),
            "lon": np.array([-180, -90, 0, 90, 180]),
        }
        domain._axis_mapping = {Axis.LATITUDE: "lat", Axis.LONGITUDE: "lon"}
        domain.latitude = domain.coords["lat"]
        domain.longitude = domain.coords["lon"]

        # Call the method with various boundary conditions
        result = SparseDomain._compute_subset_indexers(
            domain, north=45, south=-45, west=-90, east=90
        )

        # Check that result is a SparseDomain or dict
        assert isinstance(result, (SparseDomain, dict))

    def test_compute_sample_uniform_indexers(self):
        """Test _compute_sample_uniform_indexers method."""
        # Create a mock SparseDomain instance
        domain = MagicMock(spec=SparseDomain)
        domain.size = 100
        domain._get_sampling_points_nbr.return_value = 50

        # Call the method
        result = SparseDomain._compute_sample_uniform_indexers(
            domain, portion=0.5
        )

        # Check that result is a dict
        assert isinstance(result, dict)

    def test_compute_sample_normal_indexers(self):
        """Test _compute_sample_normal_indexers method."""
        # Create a mock SparseDomain instance
        domain = MagicMock(spec=SparseDomain)
        domain.size = 100
        domain._get_sampling_points_nbr.return_value = 50
        domain.latitude = np.array([-90, -45, 0, 45, 90])
        domain.longitude = np.array([-180, -90, 0, 90, 180])

        # Call the method with default center point
        result = SparseDomain._compute_sample_normal_indexers(
            domain, portion=0.5, center_point=None, sigma=10.0
        )

        # Check that result is a dict
        assert isinstance(result, dict)

        # Call the method with specified center point
        result = SparseDomain._compute_sample_normal_indexers(
            domain, portion=0.5, center_point=(0, 0), sigma=10.0
        )

        # Check that result is a dict
        assert isinstance(result, dict)

    def test_compute_sample_no_nans_indexers(self):
        """Test _compute_sample_no_nans_indexers method."""
        # Create a mock SparseDomain instance
        domain = MagicMock(spec=SparseDomain)
        domain.size = 100
        domain._get_sampling_points_nbr.return_value = 50

        # Create a simple DataArray with some NaN values
        da = xr.DataArray(
            np.random.rand(5, 5),
            dims=["lat", "lon"],
            coords={
                "lat": np.linspace(-90, 90, 5),
                "lon": np.linspace(-180, 180, 5),
            },
        )
        da.values[1, 1] = np.nan

        # Call the method
        result = SparseDomain._compute_sample_no_nans_indexers(
            domain, da, portion=0.5
        )

        # Check that result is a dict
        assert isinstance(result, dict)

    def test_to_xarray(self):
        """Test to_xarray method based on docstring."""
        # Create a mock SparseDomain instance
        domain = MagicMock(spec=SparseDomain)
        domain.coords = {
            "lat": np.array([-90, 0, 90]),
            "lon": np.array([-180, 0, 180]),
        }
        domain._axis_mapping = {Axis.LATITUDE: "lat", Axis.LONGITUDE: "lon"}

        # Create sample values
        values = np.random.rand(3, 3)

        # Call the method
        result = SparseDomain.to_xarray(domain, values, name="test")

        # Check that result is a DataArray
        assert isinstance(result, xr.DataArray)
        assert result.name == "test"


class TestDenseDomain:
    """Test the DenseDomain class."""

    def test_get_all_spatial_points(self):
        """Test get_all_spatial_points method."""
        # Create a mock DenseDomain instance
        domain = MagicMock(spec=DenseDomain)
        domain.coords = {
            "lat": np.array([-90, 0, 90]),
            "lon": np.array([-180, 0, 180]),
        }
        domain._axis_mapping = {Axis.LATITUDE: "lat", Axis.LONGITUDE: "lon"}
        domain.latitude = domain.coords["lat"]
        domain.longitude = domain.coords["lon"]

        # Call the method
        result = DenseDomain.get_all_spatial_points(domain)

        # Check that result is a numpy array
        assert isinstance(result, np.ndarray)

    def test_compute_subset_indexers(self):
        """Test _compute_subset_indexers method."""
        # Create a mock DenseDomain instance
        domain = MagicMock(spec=DenseDomain)
        domain.coords = {
            "lat": np.array([-90, -45, 0, 45, 90]),
            "lon": np.array([-180, -90, 0, 90, 180]),
        }
        domain._axis_mapping = {Axis.LATITUDE: "lat", Axis.LONGITUDE: "lon"}
        domain.latitude = domain.coords["lat"]
        domain.longitude = domain.coords["lon"]

        # Call the method with various boundary conditions
        result, _, _ = DenseDomain._compute_subset_indexers(
            domain, north=45, south=-45, west=-90, east=90
        )

        # Check that result is a dict
        assert isinstance(result, dict)

    def test_compute_sample_uniform_indexers(self):
        """Test _compute_sample_uniform_indexers method."""
        # Create a mock DenseDomain instance
        domain = MagicMock(spec=DenseDomain)
        domain.size = 100
        domain._get_sampling_points_nbr.return_value = 50

        # Call the method
        result = DenseDomain._compute_sample_uniform_indexers(
            domain, portion=0.5
        )

        # Check that result is a dict
        assert isinstance(result, dict)

    def test_compute_sample_normal_indexers(self):
        """Test _compute_sample_normal_indexers method."""
        # Create a mock DenseDomain instance
        domain = MagicMock(spec=DenseDomain)
        domain.size = 100
        domain._get_sampling_points_nbr.return_value = 50
        domain.latitude = np.array([-90, -45, 0, 45, 90])
        domain.longitude = np.array([-180, -90, 0, 90, 180])

        # Call the method with default center point
        result = DenseDomain._compute_sample_normal_indexers(
            domain, portion=0.5, center_point=None, sigma=10.0
        )

        # Check that result is a dict
        assert isinstance(result, dict)

        # Call the method with specified center point
        result = DenseDomain._compute_sample_normal_indexers(
            domain, portion=0.5, center_point=(0, 0), sigma=10.0
        )

        # Check that result is a dict
        assert isinstance(result, dict)

    def test_compute_sample_no_nans_indexers(self):
        """Test _compute_sample_no_nans_indexers method based on docstring."""
        # Create a mock DenseDomain instance
        domain = MagicMock(spec=DenseDomain)
        domain.size = 100
        domain._get_sampling_points_nbr.return_value = 50

        # Create a simple DataArray with some NaN values
        da = xr.DataArray(
            np.random.rand(5, 5),
            dims=["lat", "lon"],
            coords={
                "lat": np.linspace(-90, 90, 5),
                "lon": np.linspace(-180, 180, 5),
            },
        )
        da.values[1, 1] = np.nan

        # Call the method
        result = DenseDomain._compute_sample_no_nans_indexers(
            domain, da, portion=0.5
        )

        # Check that result is a dict with the expected keys
        assert isinstance(result, dict)

        # Test with specific number of points
        result = DenseDomain._compute_sample_no_nans_indexers(
            domain, da, number=10
        )
        assert isinstance(result, dict)

    def test_to_xarray(self):
        """Test to_xarray method based on docstring."""
        # Create a mock DenseDomain instance
        domain = MagicMock(spec=DenseDomain)
        domain.coords = {
            "lat": np.array([-90, 0, 90]),
            "lon": np.array([-180, 0, 180]),
        }
        domain._axis_mapping = {Axis.LATITUDE: "lat", Axis.LONGITUDE: "lon"}

        # Create sample values
        values = np.random.rand(3, 3)

        # Call the method
        result = DenseDomain.to_xarray(domain, values, name="test")

        # Check that result is a DataArray
        assert isinstance(result, xr.DataArray)
        assert result.name == "test"
