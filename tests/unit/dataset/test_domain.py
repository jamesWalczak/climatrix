from functools import partial
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest
import xarray as xr

from climatrix.dataset.axis import Axis, AxisType

# Import the classes and functions to be tested
from climatrix.dataset.domain import (
    DenseDomain,
    Domain,
    SamplingNaNPolicy,
    SparseDomain,
    check_is_dense,
    ensure_all_numpy_arrays,
    ensure_single_var,
    filter_out_single_value_coord,
    match_axes,
    validate_input,
    validate_spatial_axes,
)
from climatrix.exceptions import MissingAxisError
from climatrix.warnings import (
    DomainMismatchWarning,
    TooLargeSamplePortionWarning,
)


class TestSamplingNaNPolicy:

    def test_valid_policies(self):
        assert SamplingNaNPolicy.IGNORE == "ignore"
        assert SamplingNaNPolicy.RESAMPLE == "resample"
        assert SamplingNaNPolicy.RAISE == "raise"

    def test_missing_policy(self):
        with pytest.raises(
            ValueError,
            match="'invalid_policy' is not a valid SamplingNaNPolicy",
        ):
            SamplingNaNPolicy("invalid_policy")

    def test_get_method_with_string(self):
        assert SamplingNaNPolicy.get("ignore") == SamplingNaNPolicy.IGNORE
        assert SamplingNaNPolicy.get("IGNORE") == SamplingNaNPolicy.IGNORE

    def test_get_method_with_enum(self):
        policy = SamplingNaNPolicy.IGNORE
        assert SamplingNaNPolicy.get(policy) == policy

    def test_get_method_with_invalid_string(self):
        with pytest.raises(ValueError, match="Unknown Nan policy: invalid"):
            SamplingNaNPolicy.get("invalid")


class TestDomainHelperFunctions:

    def test_validate_input(self):
        da = xr.DataArray(np.random.rand(5, 5))
        validate_input(da)

        ds = xr.Dataset({"var": da})
        validate_input(ds)

    def test_ensure_single_var(self):
        da = xr.DataArray(np.random.rand(5, 5), dims=["x", "y"])
        result = ensure_single_var(da)
        assert isinstance(result, xr.DataArray)

        ds = xr.Dataset({"var": da})
        result = ensure_single_var(ds)
        assert isinstance(result, xr.DataArray)

    def test_match_axis_names(self):
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
        result = match_axes(da)
        assert AxisType.TIME in result
        assert AxisType.LATITUDE in result
        assert AxisType.LONGITUDE in result

    def test_validate_spatial_axes_if_present(self):
        axis_mapping = {AxisType.LATITUDE: "lat", AxisType.LONGITUDE: "lon"}
        validate_spatial_axes(axis_mapping)

    def test_validate_spatial_axes_if_lat_missing(self):
        axis_mapping = {AxisType.POINT: "point", AxisType.LONGITUDE: "lon"}
        with pytest.raises(ValueError, match="Dataset has no LATITUDE axis"):
            validate_spatial_axes(axis_mapping)

    def test_validate_spatial_axes_if_lon_missing(self):
        axis_mapping = {AxisType.POINT: "point", AxisType.LATITUDE: "lat"}
        with pytest.raises(ValueError, match="Dataset has no LONGITUDE axis"):
            validate_spatial_axes(axis_mapping)

    def test_check_is_dense_on_dimension_lat_lon(self):
        axis_mapping = {
            AxisType.LATITUDE: Axis("lat", np.array([-90, 0, 90]), True),
            AxisType.LONGITUDE: Axis("lon", np.array([-180, 0, 180]), True),
        }
        result = check_is_dense(axis_mapping)
        assert isinstance(result, bool)

    def test_check_is_not_dense_on_dimension_only_lat(self):
        axis_mapping = {
            AxisType.LATITUDE: Axis("lat", np.array([-90, 0, 90]), True),
            AxisType.LONGITUDE: Axis("lon", np.array([-180, 0, 180]), False),
        }
        result = check_is_dense(axis_mapping)
        assert isinstance(result, bool)

    def test_check_is_not_dense_on_dimension_only_lon(self):
        axis_mapping = {
            AxisType.LATITUDE: Axis("lat", np.array([-90, 0, 90]), False),
            AxisType.LONGITUDE: Axis("lon", np.array([-180, 0, 180]), True),
        }
        result = check_is_dense(axis_mapping)
        assert isinstance(result, bool)

    def test_ensure_all_numpy_arrays_numpy_array_passed(self):
        coords = {
            "lat": np.array([-90, 0, 90]),
            "lon": np.array([-180, 0, 180]),
        }
        coords2 = ensure_all_numpy_arrays(coords)
        assert isinstance(coords2["lat"], np.ndarray)
        assert isinstance(coords2["lon"], np.ndarray)

    def test_ensure_all_numpy_arrays_list_passed(self):
        coords = {"lat": [-90, 0, 90], "lon": [-180, 0, 180]}
        coords2 = ensure_all_numpy_arrays(coords)
        assert isinstance(coords2["lat"], np.ndarray)
        assert isinstance(coords2["lon"], np.ndarray)

    def test_filter_out_single_value_coord(self):
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

    def test_domain_factory_method_dense(self):
        dense_da = xr.DataArray(
            np.random.rand(3, 3),
            dims=["lat", "lon"],
            coords={
                "lat": np.linspace(-90, 90, 3),
                "lon": np.linspace(-180, 180, 3),
            },
        )

        with patch(
            "climatrix.dataset.domain.check_is_dense", return_value=True
        ):
            domain = Domain(dense_da)
            assert isinstance(domain, DenseDomain)

    def test_domain_factory_method_sparse(self):
        dense_da = xr.DataArray(
            np.random.rand(3, 3),
            dims=["lat", "lon"],
            coords={
                "lat": np.linspace(-90, 90, 3),
                "lon": np.linspace(-180, 180, 3),
            },
        )

        with patch(
            "climatrix.dataset.domain.check_is_dense", return_value=False
        ):
            domain = Domain(dense_da)
            assert isinstance(domain, SparseDomain)

    def test_from_lat_lon_sparse_fail_on_coord_length_mismatch(self):
        lat = np.array([-90, 0, 90])
        lon = np.array([-180, 0])
        with pytest.raises(
            ValueError,
            match="For sparse domain, lat and lon must have the same length",
        ):
            Domain.from_lat_lon(lat=lat, lon=lon, kind="sparse")

    def test_from_lat_lon_dense_coord_length_mismatch(self):
        lat = np.array([-90, 0, 90])
        lon = np.array([-180, 0])
        domain = Domain.from_lat_lon(lat=lat, lon=lon, kind="dense")
        assert isinstance(domain, DenseDomain)

    def test_from_lat_lon(self):
        lat = np.array([-90, 0, 90])
        lon = np.array([-180, 0, 180])
        domain = Domain.from_lat_lon(lat=lat, lon=lon)
        np.testing.assert_array_equal(domain.latitude.values, lat)
        np.testing.assert_array_equal(domain.longitude.values, lon)

    def test_coordinate_properties(self):
        domain = MagicMock(spec=Domain)
        domain._axes = {
            AxisType.LATITUDE: Axis(
                name="lat", is_dimension=True, values=np.array([-90, 0, 90])
            ),
            AxisType.LONGITUDE: Axis(
                name="lon", is_dimension=True, values=np.array([-180, 0, 180])
            ),
            AxisType.TIME: Axis(
                name="time",
                is_dimension=True,
                values=np.array(
                    ["2020-01-01", "2020-01-02"], dtype="datetime64"
                ),
            ),
            AxisType.POINT: Axis(
                name="point", is_dimension=True, values=np.array([1, 2, 3])
            ),
        }

        np.testing.assert_array_equal(
            Domain.latitude.__get__(domain).values,
            domain._axes[AxisType.LATITUDE].values,
        )
        np.testing.assert_array_equal(
            Domain.longitude.__get__(domain).values,
            domain._axes[AxisType.LONGITUDE].values,
        )
        np.testing.assert_array_equal(
            Domain.time.__get__(domain).values,
            domain._axes[AxisType.TIME].values,
        )
        np.testing.assert_array_equal(
            Domain.point.__get__(domain).values,
            domain._axes[AxisType.POINT].values,
        )

    def test_size_methods(self):
        domain = MagicMock(spec=Domain)
        domain.get_size = partial(Domain.get_size, domain)
        domain._axes = {
            AxisType.LATITUDE: Axis(
                name="lat", is_dimension=True, values=np.array([-90, 0, 90])
            ),
            AxisType.LONGITUDE: Axis(
                name="lon", is_dimension=True, values=np.array([-180, 0, 180])
            ),
        }
        assert domain.get_size(AxisType.LATITUDE) == 3

        domain.get_size.side_effect = lambda axis: len(domain._axes[axis])
        assert Domain.size.__get__(domain) > 0

    def test_is_dynamic_false_on_missing_time_coordinate(self):
        domain = MagicMock(spec=Domain)
        domain._axes = {
            AxisType.LATITUDE: Axis(
                name="lat", is_dimension=True, values=np.array([-90, 0, 90])
            ),
            AxisType.LONGITUDE: Axis(
                name="lon", is_dimension=True, values=np.array([-180, 0, 180])
            ),
        }
        type(domain).time = PropertyMock(side_effect=MissingAxisError)
        assert not Domain.is_dynamic.__get__(domain)

    def test_is_dynamic_false_on_single_time_coordinate(self):
        domain = MagicMock(spec=Domain)
        domain._axis_mapping = {
            AxisType.LATITUDE: "lat",
            AxisType.LONGITUDE: "lon",
        }
        domain.time.name = "time"
        domain.time = np.array(["2020-01-01"], dtype="datetime64")
        domain.get_size = partial(Domain.get_size, domain)

        assert not Domain.is_dynamic.__get__(domain)

    def test_is_dynamic_true_on_multiple_time_coordinates(self):
        domain = MagicMock(spec=Domain)
        domain._axis_mapping = {
            AxisType.LATITUDE: "lat",
            AxisType.LONGITUDE: "lon",
            AxisType.TIME: "time",
        }
        domain.get_size = partial(Domain.get_size, domain)

        domain._axes = {
            AxisType.TIME: Axis(
                name="time",
                is_dimension=True,
                values=np.array(
                    ["2020-01-01", "2020-01-02"], dtype="datetime64"
                ),
            ),
        }

        assert Domain.is_dynamic.__get__(domain)

    def test_equality_for_time_axis(self):
        domain1 = MagicMock(spec=Domain)
        domain1._axes = {
            AxisType.LATITUDE: Axis("lat", np.array([-90, 0, 90]), True),
            AxisType.LONGITUDE: Axis("lon", np.array([-180, 0, 180]), True),
            AxisType.TIME: Axis(
                "time",
                np.array(["2020-01-01", "2020-01-02"], dtype="datetime64"),
                True,
            ),
        }

        domain2 = MagicMock(spec=Domain)
        domain2._axes = {
            AxisType.LATITUDE: Axis("lat", np.array([-90, 0, 90]), True),
            AxisType.LONGITUDE: Axis("lon", np.array([-180, 0, 180]), True),
            AxisType.TIME: Axis(
                "time",
                np.array(["2020-01-01", "2020-01-02"], dtype="datetime64"),
                True,
            ),
        }

        assert Domain.__eq__(domain1, domain2)

    def test_equality_valid(self):
        domain1 = MagicMock(spec=Domain)
        domain1._axes = {
            AxisType.LATITUDE: Axis("lat", np.array([-90, 0, 90]), True),
            AxisType.LONGITUDE: Axis("lon", np.array([-180, 0, 180]), True),
        }

        domain2 = MagicMock(spec=Domain)
        domain2._axes = {
            AxisType.LATITUDE: Axis("lat", np.array([-90, 0, 90]), True),
            AxisType.LONGITUDE: Axis("lon", np.array([-180, 0, 180]), True),
        }

        assert Domain.__eq__(domain1, domain2)

    def test_equality_false_on_different_coord_values(self):
        domain1 = MagicMock(spec=Domain)
        domain1._axes = {
            AxisType.LATITUDE: Axis("lat", np.array([-90, 0, 90]), True),
            AxisType.LONGITUDE: Axis("lon", np.array([-180, 0, 180]), True),
        }
        domain1.latitude = domain1._axes[AxisType.LATITUDE]
        domain1.longitude = domain1._axes[AxisType.LONGITUDE]

        domain2 = MagicMock(spec=Domain)
        domain2._axes = {
            AxisType.LATITUDE: Axis("lat", np.array([-45, 0, 45]), True),
            AxisType.LONGITUDE: Axis("lon", np.array([-180, 0, 180]), True),
        }
        domain2.latitude = domain2._axes[AxisType.LATITUDE]
        domain2.longitude = domain2._axes[AxisType.LONGITUDE]
        assert not Domain.__eq__(domain1, domain2)

    def test_equality_false_on_different_coords(self):
        domain1 = MagicMock(spec=Domain)
        domain1._axes = {
            AxisType.LATITUDE: Axis("lat", np.array([-90, 0, 90]), True),
            AxisType.LONGITUDE: Axis("lon", np.array([-180, 0, 180]), True),
        }

        domain2 = MagicMock(spec=Domain)
        domain2._axes = {
            AxisType.POINT: Axis("point", np.array([-45, 0, 45]), True),
        }

    def test_equality_false_on_non_domain_type(self):
        domain1 = MagicMock(spec=Domain)

        assert not Domain.__eq__(domain1, "not a domain")

    def test_sampling_points_calculation_portion_passed(self):
        domain = MagicMock(spec=Domain)
        domain.size = 100

        assert (
            Domain._get_sampling_points_nbr(domain, portion=0.5, number=None)
            == 50
        )

    def test_sampling_points_calculation_number_passed(self):
        domain = MagicMock(spec=Domain)
        domain.size = 100
        assert (
            Domain._get_sampling_points_nbr(domain, portion=None, number=30)
            == 30
        )

    def test_sampling_points_calculation_fail_on_portion_and_number_passed(
        self,
    ):
        domain = MagicMock(spec=Domain)

        with pytest.raises(
            ValueError,
            match="Either portion or number must be provided, but not both",
        ):
            Domain._get_sampling_points_nbr(domain, portion=0.5, number=30)

    def test_sampling_points_calculation_fail_on_missing_portion_and_number(
        self,
    ):
        domain = MagicMock(spec=Domain)

        with pytest.raises(
            ValueError, match="Either portion or number must be provided"
        ):
            Domain._get_sampling_points_nbr(domain, portion=None, number=None)

    def test_sampling_points_calculation_warn_on_too_large_portion(self):
        domain = MagicMock(spec=Domain)

        with pytest.warns(TooLargeSamplePortionWarning):
            Domain._get_sampling_points_nbr(domain, portion=1.9, number=None)

    def test_sampling_points_calculation_warn_on_too_large_number(self):
        domain = MagicMock(spec=Domain)
        domain.size = 100

        with pytest.warns(TooLargeSamplePortionWarning):
            Domain._get_sampling_points_nbr(domain, portion=None, number=101)


class TestSparseDomain:

    def test_get_all_spatial_points(self):
        domain = MagicMock(spec=SparseDomain)
        domain.coords = {
            AxisType.LATITUDE: Axis("lat", np.array([-90, 0, 90]), True),
            AxisType.LONGITUDE: Axis("lon", np.array([-180, 0, 180]), True),
        }
        domain.latitude = domain._axes[AxisType.LATITUDE]
        domain.longitude = domain._axes[AxisType.LONGITUDE]

        result = SparseDomain.get_all_spatial_points(domain)

        assert isinstance(result, np.ndarray)

    def test_compute_subset_indexers_valid_types(self):
        domain = MagicMock(spec=SparseDomain)
        domain._axes = {
            AxisType.LATITUDE: Axis(
                "lat", np.array([-90, -45, 0, 45, 90]), True
            ),
            AxisType.LONGITUDE: Axis(
                "lon", np.array([-180, -90, 0, 90, 180]), True
            ),
        }
        domain.latitude = domain._axes[AxisType.LATITUDE]
        domain.longitude = domain._axes[AxisType.LONGITUDE]

        result = SparseDomain._compute_subset_indexers(
            domain, north=45, south=-45, west=-90, east=90
        )
        assert isinstance(result, tuple)
        assert isinstance(result[0], dict)

    def test_compute_subset_indexers_valid_points(self):
        domain = MagicMock(spec=SparseDomain)
        domain._axes = {
            AxisType.LATITUDE: Axis(
                "lat", np.array([-90, -45, 0, 45, 90]), True
            ),
            AxisType.LONGITUDE: Axis(
                "lon", np.array([-180, -90, 0, 90, 180]), True
            ),
        }
        domain.latitude = domain._axes[AxisType.LATITUDE]
        domain.longitude = domain._axes[AxisType.LONGITUDE]

        result = SparseDomain._compute_subset_indexers(
            domain, north=45, south=-45, west=-90, east=90
        )
        assert isinstance(result, tuple)
        assert list(next(iter(result[0].values()))) == [1, 2, 3]

    def test_compute_sample_uniform_indexers(self):
        domain = MagicMock(spec=SparseDomain)
        domain.size = 100
        domain._get_sampling_points_nbr.return_value = 50
        domain.point.size = 100

        result = SparseDomain._compute_sample_uniform_indexers(
            domain, portion=0.5
        )

        assert isinstance(result, dict)

    def test_compute_sample_normal_indexers(self):
        domain = MagicMock(spec=SparseDomain)
        domain.size = 100
        domain.point = Axis("point", np.array([1, 2, 3, 4, 5]), True)
        domain._get_sampling_points_nbr.return_value = 50
        domain.latitude = Axis("lat", np.array([-90, -45, 0, 45, 90]), True)
        domain.longitude = Axis("lon", np.array([-180, -90, 0, 90, 180]), True)

        result = SparseDomain._compute_sample_normal_indexers(
            domain, portion=0.5, center_point=None, sigma=10.0
        )

        assert isinstance(result, dict)

        result = SparseDomain._compute_sample_normal_indexers(
            domain, portion=0.5, center_point=(0, 0), sigma=10.0
        )

        assert isinstance(result, dict)

    def test_compute_sample_no_nans_indexers_return_no_nan(self):
        da = xr.DataArray(
            np.random.rand(500, 600),
            dims=["lat", "lon"],
            coords={
                "lat": np.linspace(-90, 90, 500),
                "lon": np.linspace(-180, 180, 600),
            },
        )
        idx_lat = np.random.choice(len(da.lat), size=100, replace=True)
        idx_lon = np.random.choice(len(da.lon), size=100, replace=True)

        da.values[idx_lat, idx_lon] = np.nan
        domain = Domain(da)

        result = DenseDomain._compute_sample_no_nans_indexers(
            domain, da, portion=0.5
        )
        values = da.isel(result).values
        assert np.isnan(values).sum() == 0


class TestDenseDomain:

    def test_compute_sample_uniform_indexers(self):
        domain = MagicMock(spec=DenseDomain)
        domain.size = 100
        domain._get_sampling_points_nbr.return_value = 50
        domain.latitude = Axis("lat", np.array([-90, -45, 0, 45, 90]), True)
        domain.longitude = Axis("lon", np.array([-180, -90, 0, 90, 180]), True)

        result = DenseDomain._compute_sample_uniform_indexers(
            domain, portion=0.5
        )

        assert isinstance(result, dict)

    def test_compute_sample_normal_indexers(self):
        domain = MagicMock(spec=DenseDomain)
        domain.size = 100
        domain._get_sampling_points_nbr.return_value = 50
        domain.latitude = Axis(
            "lat", is_dimension=True, values=np.array([-90, -45, 0, 45, 90])
        )
        domain.longitude = Axis(
            "lon", is_dimension=True, values=np.array([-180, -90, 0, 90, 180])
        )

        result = DenseDomain._compute_sample_normal_indexers(
            domain, portion=0.5, center_point=None, sigma=10.0
        )

        assert isinstance(result, dict)

        result = DenseDomain._compute_sample_normal_indexers(
            domain, portion=0.5, center_point=(0, 0), sigma=10.0
        )

        assert isinstance(result, dict)

    def test_compute_sample_no_nans_indexers_return_no_nan(self):
        da = xr.DataArray(
            np.random.rand(500, 600),
            dims=["lat", "lon"],
            coords={
                "lat": np.linspace(-90, 90, 500),
                "lon": np.linspace(-180, 180, 600),
            },
        )
        idx_lat = np.random.choice(len(da.lat), size=100, replace=True)
        idx_lon = np.random.choice(len(da.lon), size=100, replace=True)

        da.values[idx_lat, idx_lon] = np.nan
        domain = Domain(da)

        result = DenseDomain._compute_sample_no_nans_indexers(
            domain, da, portion=0.5
        )
        values = da.isel(result).values
        assert np.isnan(values).sum() == 0

    def test_equality_missing_coordinates(self):
        domain1 = MagicMock(spec=Domain)
        domain1._axes = {
            AxisType.LATITUDE: Axis("lat", np.array([-90, 0, 90]), True),
        }
        domain1.latitude = domain1._axes[AxisType.LATITUDE]
        domain2 = MagicMock(spec=Domain)
        domain2._axes = {
            AxisType.LATITUDE: Axis("lat", np.array([-90, 0, 90]), True),
            AxisType.LONGITUDE: Axis("lon", np.array([-180, 0, 180]), True),
        }
        domain2.latitude = domain2._axes[AxisType.LATITUDE]
        domain2.longitude = domain2._axes[AxisType.LONGITUDE]
        with pytest.warns(DomainMismatchWarning, match="Domain mismatch:"):
            assert not Domain.__eq__(domain1, domain2)

    def test_equality_extra_coordinates(self):
        domain1 = MagicMock(spec=Domain)
        domain1._axes = {
            AxisType.LATITUDE: Axis("lat", np.array([-90, 0, 90])),
            AxisType.LONGITUDE: Axis("lon", np.array([-180, 0, 180])),
        }
        domain1.latitude = domain1._axes[AxisType.LATITUDE]
        domain1.longitude = domain1._axes[AxisType.LONGITUDE]
        domain2 = MagicMock(spec=Domain)
        domain2._axes = {
            AxisType.LATITUDE: Axis("lat", np.array([-90, 0, 90]), True),
        }
        domain2.latitude = domain2._axes[AxisType.LATITUDE]
        with pytest.warns(DomainMismatchWarning, match="Domain mismatch:"):
            assert not Domain.__eq__(domain1, domain2)

    def test_equality_non_domain_object(self):
        domain1 = MagicMock(spec=Domain)
        non_domain_object = "not_a_domain"
        assert not Domain.__eq__(domain1, non_domain_object)

    def test_equality_nan_values(self):
        domain1 = MagicMock(spec=Domain)
        domain1._axes = {
            AxisType.LATITUDE: Axis("lat", np.array([-90, np.nan, 90]), True),
            AxisType.LONGITUDE: Axis("lon", np.array([-180, 0, 180]), True),
        }
        domain1.latitude = domain1._axes[AxisType.LATITUDE]
        domain1.longitude = domain1._axes[AxisType.LONGITUDE]

        domain2 = MagicMock(spec=Domain)
        domain2._axes = {
            AxisType.LATITUDE: Axis("lat", np.array([-90, np.nan, 90]), True),
            AxisType.LONGITUDE: Axis("lon", np.array([-180, 0, 180]), True),
        }
        domain2.latitude = domain2._axes[AxisType.LATITUDE]
        domain2.longitude = domain2._axes[AxisType.LONGITUDE]
        assert Domain.__eq__(domain1, domain2)

    def test_equality_nan_values_different(self):
        domain1 = MagicMock(spec=Domain)
        domain1._axes = {
            AxisType.LATITUDE: Axis("lat", np.array([-90, np.nan, 90]), True),
            AxisType.LONGITUDE: Axis("lon", np.array([-180, 0, 180]), True),
        }
        domain1.latitude = domain1._axes[AxisType.LATITUDE]
        domain1.longitude = domain1._axes[AxisType.LONGITUDE]

        domain2 = MagicMock(spec=Domain)
        domain2._axes = {
            AxisType.LATITUDE: Axis("lat", np.array([-90, 0, 90]), True),
            AxisType.LONGITUDE: Axis("lon", np.array([-180, 0, 180]), True),
        }
        domain2.latitude = domain2._axes[AxisType.LATITUDE]
        domain2.longitude = domain2._axes[AxisType.LONGITUDE]
        assert not Domain.__eq__(domain1, domain2)
