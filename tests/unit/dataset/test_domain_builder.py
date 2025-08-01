"""
Tests for the Domain.from_axes() builder pattern functionality.
"""

import numpy as np
import pytest
import xarray as xr

from climatrix.dataset.domain import Domain, DomainBuilder


class TestDomainBuilder:
    """Test cases for the DomainBuilder class and Domain.from_axes() method."""

    def test_domain_from_axes_creates_builder(self):
        """Test that Domain.from_axes() creates a DomainBuilder instance."""
        builder = Domain.from_axes()
        assert isinstance(builder, DomainBuilder)

    def test_basic_sparse_domain_creation(self):
        """Test creating a basic sparse domain with lat/lon."""
        domain = (
            Domain.from_axes()
            .lat(latitude=[1, 2, 3])
            .lon(longitude=[4, 5, 6])
            .sparse()
        )
        assert domain.is_sparse
        assert len(domain.latitude.values) == 3
        assert len(domain.longitude.values) == 3
        np.testing.assert_array_equal(domain.latitude.values, [1, 2, 3])
        np.testing.assert_array_equal(domain.longitude.values, [4, 5, 6])

    def test_basic_dense_domain_creation(self):
        """Test creating a basic dense domain with lat/lon."""
        domain = (
            Domain.from_axes()
            .lat(latitude=[1, 2])
            .lon(longitude=[3, 4])
            .dense()
        )
        assert not domain.is_sparse
        assert len(domain.latitude.values) == 2
        assert len(domain.longitude.values) == 2
        assert domain.shape == (2, 2)

    def test_vertical_axis_in_sparse_domain(self):
        """Test adding vertical axis to sparse domain."""
        domain = (
            Domain.from_axes()
            .vertical(depth=slice(10, 50, 10))
            .lat(latitude=[1, 2])
            .lon(longitude=[3, 4])
            .sparse()
        )

        assert domain.has_axis("vertical")
        assert domain.vertical.name == "depth"
        np.testing.assert_array_equal(domain.vertical.values, [10, 20, 30, 40])

    def test_time_axis_in_domain(self):
        """Test adding time axis to domain."""
        domain = (
            Domain.from_axes()
            .lat(latitude=[1, 2])
            .lon(longitude=[3, 4])
            .time(time=["2020-01-01", "2020-01-02"])
            .dense()
        )

        assert domain.has_axis("time")
        assert domain.time.name == "time"

    def test_slice_to_array_conversion(self):
        """Test that slices are properly converted to arrays."""
        builder = DomainBuilder()
        result = builder._convert_slice_to_array(slice(1, 10, 2))
        expected = np.array([1, 3, 5, 7, 9])
        np.testing.assert_array_equal(result, expected)

    def test_custom_axis_names(self):
        """Test using custom names for axes."""
        domain = (
            Domain.from_axes()
            .vertical(pressure=[1000, 850, 500])
            .lat(latitude=[0, 1])
            .lon(longitude=[2, 3])
            .sparse()
        )

        assert domain.vertical.name == "pressure"
        np.testing.assert_array_equal(domain.vertical.values, [1000, 850, 500])

    def test_method_chaining(self):
        """Test that methods can be chained."""
        builder = (
            Domain.from_axes()
            .lat(latitude=[1])
            .lon(longitude=[2])
            .vertical(depth=[10])
            .time(time=["2020-01-01"])
        )

        assert len(builder._axes_config) == 4

    def test_sparse_validation_mismatched_coordinates(self):
        """Test validation for sparse domains with mismatched coordinate lengths."""
        with pytest.raises(
            ValueError,
            match="latitude and longitude arrays must have the same length",
        ):
            Domain.from_axes().lat(latitude=[1, 2]).lon(
                longitude=[3, 4, 5]
            ).sparse()

    def test_sparse_validation_missing_latitude(self):
        """Test validation for sparse domains missing latitude."""
        with pytest.raises(ValueError, match="Latitude axis is required"):
            Domain.from_axes().lon(longitude=[1, 2]).sparse()

    def test_sparse_validation_missing_longitude(self):
        """Test validation for sparse domains missing longitude."""
        with pytest.raises(ValueError, match="Longitude axis is required"):
            Domain.from_axes().lat(latitude=[1, 2]).sparse()

    def test_multiple_kwargs_error(self):
        """Test that providing multiple kwargs to a method raises an error."""
        with pytest.raises(
            ValueError, match="Exactly one keyword argument must be provided"
        ):
            Domain.from_axes().vertical(depth=[1], pressure=[2])

    def test_slice_without_step_error(self):
        """Test that slices without step raise an error."""
        with pytest.raises(ValueError, match="Slice step must be specified"):
            Domain.from_axes().lat(latitude=slice(1, 10)).lon(
                longitude=[1]
            ).sparse()

    def test_equivalence_with_from_lat_lon(self):
        """Test that the new method produces equivalent results to from_lat_lon."""
        lat_vals = [1, 2, 3]
        lon_vals = [4, 5, 6]

        # Create using old method
        domain_old = Domain.from_lat_lon(
            lat=np.array(lat_vals), lon=np.array(lon_vals), kind="sparse"
        )

        # Create using new method
        domain_new = (
            Domain.from_axes().lat(lat=lat_vals).lon(lon=lon_vals).sparse()
        )

        assert domain_old.is_sparse == domain_new.is_sparse
        np.testing.assert_array_equal(
            domain_old.latitude.values, domain_new.latitude.values
        )
        np.testing.assert_array_equal(
            domain_old.longitude.values, domain_new.longitude.values
        )

    def test_full_example_from_issue(self):
        """Test the exact example from the issue description."""
        # Domain.from_axes().vertical(depth=slice(10, 100, 1)).lat(latitude=[1,2,3,4]).lon(...).time(....).sparse()
        domain = (
            Domain.from_axes()
            .vertical(depth=slice(10, 100, 30))
            .lat(latitude=[1, 2, 3, 4])
            .lon(longitude=[5, 6, 7, 8])
            .time(time=["2020-01-01", "2020-01-02"])
            .sparse()
        )

        assert domain.is_sparse
        assert domain.has_axis("vertical")
        assert domain.has_axis("time")
        assert domain.vertical.name == "depth"
        assert domain.time.name == "time"
        assert len(domain.latitude.values) == 4
        assert len(domain.longitude.values) == 4
