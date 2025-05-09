from datetime import datetime

import numpy as np
import pytest
import xarray as xr

from climatrix.dataset.base import BaseClimatrixDataset
from climatrix.exceptions import LongitudeConventionMismatch
from climatrix.types import Latitude, Longitude


@pytest.fixture
def static_sample_dataarray():
    time = np.array(
        [
            "2000-01-01",
        ],
        dtype="datetime64",
    )
    lat = np.array([-45.0, 0.0, 45.0])
    lon = np.array([0.0, 180.0, 360.0])
    data = np.arange(9).reshape(1, 3, 3)
    return xr.DataArray(
        data,
        coords=[("time", time), ("lat", lat), ("lon", lon)],
        name="temperature",
    )


@pytest.fixture
def dynamic_sample_dataarray():
    time = np.array(
        ["2000-01-01", "2000-01-02", "2000-01-03"], dtype="datetime64"
    )
    lat = np.array([-45.0, 0.0, 45.0])
    lon = np.array([0.0, 180.0, 360.0])
    data = np.arange(27).reshape(3, 3, 3)
    return xr.DataArray(
        data,
        coords=[("time", time), ("lat", lat), ("lon", lon)],
        name="temperature",
    )


@pytest.fixture
def sample_static_dataset(static_sample_dataarray):
    return BaseClimatrixDataset(static_sample_dataarray)


@pytest.fixture
def sample_dynamic_dataset(dynamic_sample_dataarray):
    return BaseClimatrixDataset(dynamic_sample_dataarray)


class TestBaseClimatrixDataset:

    def test_subset_within_bounds_returns_smaller_area(
        self, sample_static_dataset
    ):
        result = sample_static_dataset.subset(
            north=10, south=-10, west=90, east=270
        )
        assert isinstance(result, BaseClimatrixDataset)
        assert result.da.lat.size <= sample_static_dataset.da.lat.size
        assert result.da.lon.size <= sample_static_dataset.da.lon.size
        assert (result.da.lat >= -10).all()
        assert (result.da.lat <= 10).all()

    def test_subset_outside_bounds_returns_empty(self, sample_static_dataset):
        result = sample_static_dataset.subset(
            north=-60, south=-90, west=380, east=420
        )
        assert result.da.size == 0

    def test_subset_raise_error_on_lon_convention_mismatch_positive_only(
        self, sample_static_dataset
    ):
        with pytest.raises(
            LongitudeConventionMismatch,
            match="The dataset is in positive-only convention*",
        ):
            sample_static_dataset.subset(
                north=10, south=-10, west=-80, east=80
            )

    def test_subset_raise_error_on_lon_convention_mismatch_signed(self):
        time = np.array(
            ["2000-01-01", "2000-01-02", "2000-01-03"], dtype="datetime64"
        )
        lat = np.array([-45.0, 0.0, 45.0])
        lon = np.array([-180, 0.0, 180.0])
        data = np.arange(27).reshape(3, 3, 3)
        dataset = BaseClimatrixDataset(
            xr.DataArray(
                data,
                coords=[("time", time), ("lat", lat), ("lon", lon)],
                name="temperature",
            )
        )

        with pytest.raises(
            LongitudeConventionMismatch,
            match="The dataset is in signed-longitude convention*",
        ):
            dataset.subset(north=10, south=-10, west=80, east=280)

    def test_to_signed_longitude_converts_range(self, sample_static_dataset):
        result = sample_static_dataset.to_signed_longitude()
        assert (result.da.lon <= 180).all()
        assert (result.da.lon >= -180).all()

    def test_to_positive_longitude_converts_range(self, sample_static_dataset):
        result = sample_static_dataset.to_positive_longitude()
        assert (result.da.lon >= 0).all()
        assert (result.da.lon <= 360).all()

    def test_mask_nan_propagates_values(self, sample_dynamic_dataset):
        masked_data = sample_dynamic_dataset.da.copy(deep=True)
        masked_data.values = masked_data.values.astype(float)
        masked_data[0, 0, 0] = np.nan
        masked_data[1, 2, 1] = np.nan
        source = BaseClimatrixDataset(masked_data)
        result = sample_dynamic_dataset.mask_nan(source)
        assert np.isnan(result.da[0, 0, 0])
        assert np.isnan(result.da[1, 2, 1])

    def test_time_selection_returns_expected_shape(
        self, sample_dynamic_dataset
    ):
        dt = datetime(2000, 1, 2)
        result = sample_dynamic_dataset.time(dt)
        assert "time" not in result.da.dims
        assert result.da.shape == (3, 3)

    def test_time_selection_returns_expected_steps(
        self, sample_dynamic_dataset
    ):
        dt = [datetime(2000, 1, 2), datetime(2000, 1, 3)]
        result = sample_dynamic_dataset.time(dt)
        assert "time" in result.da.dims
        assert result.da.shape == (2, 3, 3)
        assert result.da.time.values[0] == dt[0]
        assert result.da.time.values[1] == dt[1]

    def test_itime_indexing_returns_correct_slice(
        self, sample_dynamic_dataset
    ):
        result = sample_dynamic_dataset.itime(slice(0, 2))
        assert result.da.time.size == 2
        assert str(result.da.time.values[0])[:10] == "2000-01-01"

    def test_sample_uniform_by_number_reduces_count(
        self, sample_static_dataset
    ):
        result = sample_static_dataset.sample_uniform(number=5)
        assert result.da.count().item() == 5

    def test_sample_uniform_by_portion_selects_correct_fraction(
        self, sample_static_dataset
    ):
        portion = 0.25
        total_points = sample_static_dataset.da.size
        result = sample_static_dataset.sample_uniform(portion=portion)
        sampled_points = result.da.count().item()
        assert sampled_points == int(total_points * portion)

    def test_sample_normal_centers_correctly(self, sample_static_dataset):
        center = (Longitude(180.0), Latitude(0.0))
        result = sample_static_dataset.sample_normal(
            center_point=center, sigma=1.0, number=5
        )
        assert isinstance(result, BaseClimatrixDataset)
        non_nan_coords = result.da.where(~np.isnan(result.da), drop=True)
        assert non_nan_coords.lat.median() == pytest.approx(0.0, abs=15.0)
        assert non_nan_coords.lon.median() == pytest.approx(180.0, abs=15.0)

    def test_reconstruct_with_idw_returns_domain_shape(
        self, sample_static_dataset
    ):
        domain = sample_static_dataset.domain
        result = sample_static_dataset.reconstruct(target=domain, method="idw")
        assert result.da.shape == sample_static_dataset.da.squeeze().shape

    def test_reconstruct_invalid_method_raises(self, sample_static_dataset):
        with pytest.raises(ValueError):
            sample_static_dataset.reconstruct(
                target=sample_static_dataset.domain, method="cubic"
            )
