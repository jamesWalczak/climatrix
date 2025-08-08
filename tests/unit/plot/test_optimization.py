from unittest.mock import Mock, patch

import numpy as np

from climatrix.plot.core import Plot


class TestPlotOptimization:
    def setup_method(self):
        self.mock_dataset = Mock()
        self.mock_dataset.domain = Mock()
        self.mock_dataset.domain.is_sparse = False
        self.mock_dataset.domain.has_axis.return_value = False

        self.lats = np.linspace(-90, 90, 100)
        self.lons = np.linspace(-180, 180, 200)
        self.mock_dataset.domain.latitude = Mock()
        self.mock_dataset.domain.latitude.values = self.lats
        self.mock_dataset.domain.longitude = Mock()
        self.mock_dataset.domain.longitude.values = self.lons

        self.values = np.random.rand(100, 200)
        self.mock_dataset.da = Mock()
        self.mock_dataset.da.values = self.values

        self.plot = Plot(self.mock_dataset)

    def test_lod_optimization_with_roi(self):
        lat_min, lat_max = -10, 10
        lon_min, lon_max = -20, 20

        lats_opt, lons_opt, values_opt = self.plot._apply_lod_optimization(
            self.lats,
            self.lons,
            self.values,
            zoom_level=5,
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
        )

        assert np.all(lats_opt >= lat_min - 5)
        assert np.all(lats_opt <= lat_max + 5)
        assert np.all(lons_opt >= lon_min - 5)
        assert np.all(lons_opt <= lon_max + 5)

    def test_empty_roi(self):
        lat_min, lat_max = 95, 100
        lon_min, lon_max = 185, 190

        lats_opt, lons_opt, values_opt = self.plot._apply_lod_optimization(
            self.lats,
            self.lons,
            self.values,
            zoom_level=5,
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
        )

        assert len(lats_opt) > 0
        assert len(lons_opt) > 0
        assert values_opt.size > 0

    def test_lod_optimization_no_roi(self):
        lats_opt, lons_opt, values_opt = self.plot._apply_lod_optimization(
            self.lats,
            self.lons,
            self.values,
            zoom_level=3,
            lat_min=None,
            lat_max=None,
            lon_min=None,
            lon_max=None,
        )

        assert len(lats_opt) < len(self.lats)
        assert len(lons_opt) < len(self.lons)
        assert values_opt.shape[0] == len(lats_opt)
        assert values_opt.shape[1] == len(lons_opt)

    def test_zoom_level_step_mapping(self):
        lats_high, lons_high, values_high = self.plot._apply_lod_optimization(
            self.lats,
            self.lons,
            self.values,
            zoom_level=8,
            lat_min=None,
            lat_max=None,
            lon_min=None,
            lon_max=None,
        )

        lats_med, lons_med, values_med = self.plot._apply_lod_optimization(
            self.lats,
            self.lons,
            self.values,
            zoom_level=3,
            lat_min=None,
            lat_max=None,
            lon_min=None,
            lon_max=None,
        )

        lats_low, lons_low, values_low = self.plot._apply_lod_optimization(
            self.lats,
            self.lons,
            self.values,
            zoom_level=0,
            lat_min=None,
            lat_max=None,
            lon_min=None,
            lon_max=None,
        )

        assert len(lats_high) >= len(lats_med) >= len(lats_low)
        assert len(lons_high) >= len(lons_med) >= len(lons_low)

    @patch("climatrix.plot.core.LOD_THRESHOLD", new=10)
    def test_1d_data_optimization(self):
        values_1d = np.random.rand(100)

        lats_opt, lons_opt, values_opt = self.plot._apply_lod_optimization(
            self.lats,
            self.lons,
            values_1d,
            zoom_level=2,
            lat_min=None,
            lat_max=None,
            lon_min=None,
            lon_max=None,
        )

        assert len(lats_opt) < len(self.lats)
        assert len(lons_opt) < len(self.lons)
        assert values_opt.size < len(values_1d)

    def test_small_dataset_below_threshold(self):
        small_lats = np.linspace(-90, 90, 10)
        small_lons = np.linspace(-180, 180, 10)
        small_values = np.random.rand(10, 10)

        lats_opt, lons_opt, values_opt = self.plot._apply_lod_optimization(
            small_lats,
            small_lons,
            small_values,
            zoom_level=1,
            lat_min=None,
            lat_max=None,
            lon_min=None,
            lon_max=None,
        )

        np.testing.assert_array_equal(lats_opt, small_lats)
        np.testing.assert_array_equal(lons_opt, small_lons)
        np.testing.assert_array_equal(values_opt, small_values)

    def test_roi_boundary_conditions(self):
        lat_min, lat_max = -90, -80
        lon_min, lon_max = -180, -170

        lats_opt, lons_opt, values_opt = self.plot._apply_lod_optimization(
            self.lats,
            self.lons,
            self.values,
            zoom_level=5,
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
        )

        assert len(lats_opt) > 0
        assert len(lons_opt) > 0
        assert values_opt.size > 0

    def test_roi_exact_match(self):
        lat_min = self.lats[10]
        lat_max = self.lats[20]
        lon_min = self.lons[30]
        lon_max = self.lons[40]

        lats_opt, lons_opt, values_opt = self.plot._apply_lod_optimization(
            self.lats,
            self.lons,
            self.values,
            zoom_level=8,
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
        )

        assert np.min(lats_opt) <= lat_min
        assert np.max(lats_opt) >= lat_max
        assert np.min(lons_opt) <= lon_min
        assert np.max(lons_opt) >= lon_max

    def test_roi_single_point(self):
        lat_center = self.lats[50]
        lon_center = self.lons[100]
        tolerance = 0.1

        lats_opt, lons_opt, values_opt = self.plot._apply_lod_optimization(
            self.lats,
            self.lons,
            self.values,
            zoom_level=8,
            lat_min=lat_center - tolerance,
            lat_max=lat_center + tolerance,
            lon_min=lon_center - tolerance,
            lon_max=lon_center + tolerance,
        )

        assert len(lats_opt) > 0
        assert len(lons_opt) > 0

    def test_roi_with_1d_data(self):
        values_1d = np.random.rand(100)

        lat_min, lat_max = -10, 10
        lon_min, lon_max = -20, 20

        lats_opt, lons_opt, values_opt = self.plot._apply_lod_optimization(
            self.lats,
            self.lons,
            values_1d,
            zoom_level=5,
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
        )

        np.testing.assert_array_equal(lats_opt, self.lats)
        np.testing.assert_array_equal(lons_opt, self.lons)
        np.testing.assert_array_equal(values_opt, values_1d)

    def test_zero_zoom_level(self):
        lats_opt, lons_opt, values_opt = self.plot._apply_lod_optimization(
            self.lats,
            self.lons,
            self.values,
            zoom_level=0,
            lat_min=None,
            lat_max=None,
            lon_min=None,
            lon_max=None,
        )

        expected_lat_count = len(self.lats[::16])
        expected_lon_count = len(self.lons[::16])

        assert len(lats_opt) == expected_lat_count
        assert len(lons_opt) == expected_lon_count

    def test_negative_zoom_level(self):
        lats_opt, lons_opt, values_opt = self.plot._apply_lod_optimization(
            self.lats,
            self.lons,
            self.values,
            zoom_level=-1,
            lat_min=None,
            lat_max=None,
            lon_min=None,
            lon_max=None,
        )

        expected_lat_count = len(self.lats[::16])
        expected_lon_count = len(self.lons[::16])

        assert len(lats_opt) == expected_lat_count
        assert len(lons_opt) == expected_lon_count

    def test_very_high_zoom_level(self):
        lats_opt, lons_opt, values_opt = self.plot._apply_lod_optimization(
            self.lats,
            self.lons,
            self.values,
            zoom_level=20,
            lat_min=None,
            lat_max=None,
            lon_min=None,
            lon_max=None,
        )

        assert len(lats_opt) == len(self.lats)
        assert len(lons_opt) == len(self.lons)

    def test_partial_roi_parameters(self):
        lats_opt, lons_opt, values_opt = self.plot._apply_lod_optimization(
            self.lats,
            self.lons,
            self.values,
            zoom_level=5,
            lat_min=-10,
            lat_max=10,
            lon_min=None,
            lon_max=None,
        )

        assert len(lats_opt) > 0
        assert len(lons_opt) > 0

    def test_inverted_roi_bounds(self):
        lats_opt, lons_opt, values_opt = self.plot._apply_lod_optimization(
            self.lats,
            self.lons,
            self.values,
            zoom_level=5,
            lat_min=10,
            lat_max=-10,
            lon_min=20,
            lon_max=-20,
        )

        assert len(lats_opt) > 0
        assert len(lons_opt) > 0

    def test_roi_completely_outside_bounds(self):
        lat_min, lat_max = 100, 200
        lon_min, lon_max = 200, 300

        lats_opt, lons_opt, values_opt = self.plot._apply_lod_optimization(
            self.lats,
            self.lons,
            self.values,
            zoom_level=5,
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
        )

        assert len(lats_opt) > 0
        assert len(lons_opt) > 0

    def test_empty_data_arrays(self):
        empty_lats = np.array([])
        empty_lons = np.array([])
        empty_values = np.array([]).reshape(0, 0)

        lats_opt, lons_opt, values_opt = self.plot._apply_lod_optimization(
            empty_lats,
            empty_lons,
            empty_values,
            zoom_level=5,
            lat_min=None,
            lat_max=None,
            lon_min=None,
            lon_max=None,
        )

        assert len(lats_opt) == 0
        assert len(lons_opt) == 0
        assert values_opt.size == 0

    def test_single_point_data(self):
        single_lat = np.array([0.0])
        single_lon = np.array([0.0])
        single_value = np.array([[1.0]])

        lats_opt, lons_opt, values_opt = self.plot._apply_lod_optimization(
            single_lat,
            single_lon,
            single_value,
            zoom_level=5,
            lat_min=None,
            lat_max=None,
            lon_min=None,
            lon_max=None,
        )

        # Should return the single point (below threshold)
        assert len(lats_opt) == 1
        assert len(lons_opt) == 1
        assert values_opt.size == 1

    def test_nan_values_handling(self):
        """Test handling of NaN values in data"""
        # Create data with NaN values
        values_with_nan = self.values.copy()
        values_with_nan[10:20, 30:40] = np.nan

        lats_opt, lons_opt, values_opt = self.plot._apply_lod_optimization(
            self.lats,
            self.lons,
            values_with_nan,
            zoom_level=5,
            lat_min=None,
            lat_max=None,
            lon_min=None,
            lon_max=None,
        )

        # Should handle NaN values gracefully
        assert len(lats_opt) > 0
        assert len(lons_opt) > 0
        assert values_opt.size > 0

    def test_infinite_values_handling(self):
        """Test handling of infinite values in data"""
        # Create data with infinite values
        values_with_inf = self.values.copy()
        values_with_inf[10:20, 30:40] = np.inf
        values_with_inf[50:60, 70:80] = -np.inf

        lats_opt, lons_opt, values_opt = self.plot._apply_lod_optimization(
            self.lats,
            self.lons,
            values_with_inf,
            zoom_level=5,
            lat_min=None,
            lat_max=None,
            lon_min=None,
            lon_max=None,
        )

        # Should handle infinite values gracefully
        assert len(lats_opt) > 0
        assert len(lons_opt) > 0
        assert values_opt.size > 0

    def test_large_dataset_performance(self):
        """Test optimization with very large dataset"""
        # Create a large dataset
        large_lats = np.linspace(-90, 90, 500)
        large_lons = np.linspace(-180, 180, 1000)
        large_values = np.random.rand(500, 1000)  # 500k points

        lats_opt, lons_opt, values_opt = self.plot._apply_lod_optimization(
            large_lats,
            large_lons,
            large_values,
            zoom_level=1,
            lat_min=None,
            lat_max=None,
            lon_min=None,
            lon_max=None,
        )

        # Should significantly reduce the data size
        assert len(lats_opt) < len(large_lats)
        assert len(lons_opt) < len(large_lons)
        assert values_opt.size < large_values.size

    def test_data_type_preservation(self):
        """Test that data types are preserved after optimization"""
        # Test with different data types
        float32_values = self.values.astype(np.float32)

        lats_opt, lons_opt, values_opt = self.plot._apply_lod_optimization(
            self.lats,
            self.lons,
            float32_values,
            zoom_level=5,
            lat_min=None,
            lat_max=None,
            lon_min=None,
            lon_max=None,
        )

        # Data should maintain its structure (though type might change due to slicing)
        assert isinstance(lats_opt, np.ndarray)
        assert isinstance(lons_opt, np.ndarray)
        assert isinstance(values_opt, np.ndarray)

    def test_step_size_boundaries(self):
        """Test all step size boundaries in zoom level mapping"""
        test_cases = [
            (10, 1),  # zoom_level >= 8
            (8, 1),  # zoom_level >= 8 (boundary)
            (7, 2),  # zoom_level >= 5
            (5, 2),  # zoom_level >= 5 (boundary)
            (4, 4),  # zoom_level >= 3
            (3, 4),  # zoom_level >= 3 (boundary)
            (2, 8),  # zoom_level >= 1
            (1, 8),  # zoom_level >= 1 (boundary)
            (0, 16),  # else case
        ]

        for zoom_level, expected_step in test_cases:
            lats_opt, lons_opt, values_opt = self.plot._apply_lod_optimization(
                self.lats,
                self.lons,
                self.values,
                zoom_level=zoom_level,
                lat_min=None,
                lat_max=None,
                lon_min=None,
                lon_max=None,
            )

            expected_lat_count = len(self.lats[::expected_step])
            expected_lon_count = len(self.lons[::expected_step])

            assert (
                len(lats_opt) == expected_lat_count
            ), f"Zoom {zoom_level}, step {expected_step}"
            assert (
                len(lons_opt) == expected_lon_count
            ), f"Zoom {zoom_level}, step {expected_step}"
