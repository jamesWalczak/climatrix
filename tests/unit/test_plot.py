import importlib.resources
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
import xarray as xr

from climatrix.dataset.base import BaseClimatrixDataset
from climatrix.plot.core import Plot


@pytest.fixture
def sample_dense_dataset():
    lat = np.array([-45.0, 0.0, 45.0])
    lon = np.array([0.0, 180.0, 360.0])
    time = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64")
    data = np.random.rand(2, 3, 3)

    da = xr.DataArray(
        data,
        coords=[("time", time), ("lat", lat), ("lon", lon)],
        name="temperature",
    )
    return BaseClimatrixDataset(da)


@pytest.fixture
def sample_sparse_dataset():
    points = np.array([0, 1, 2])
    lat = np.array([-45.0, 0.0, 45.0])
    lon = np.array([0.0, 180.0, 360.0])
    time = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64")
    data = np.random.rand(2, 3)

    da = xr.DataArray(
        data,
        coords={
            "time": time,
            "point": points,
            "lat": ("point", lat),
            "lon": ("point", lon),
        },
        dims=["time", "point"],
        name="temperature",
    )
    return BaseClimatrixDataset(da)


@pytest.fixture
def sample_dataset_with_vertical():
    lat = np.array([-45.0, 0.0, 45.0])
    lon = np.array([0.0, 180.0, -180.0])
    time = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64")
    level = np.array([850, 500, 200])
    data = np.random.rand(2, 3, 3, 3)

    da = xr.DataArray(
        data,
        coords=[("time", time), ("level", level), ("lat", lat), ("lon", lon)],
        name="temperature",
    )
    return BaseClimatrixDataset(da)


@pytest.fixture
def sample_static_dense_dataset():
    lat = np.array([-45.0, 0.0, 45.0])
    lon = np.array([0.0, 180.0, -180])
    data = np.random.rand(3, 3)

    da = xr.DataArray(
        data,
        coords=[("lat", lat), ("lon", lon)],
        name="temperature",
    )
    return BaseClimatrixDataset(da)


class TestPlotInitialization:

    def test_plot_initialization_with_dataset(self, sample_dense_dataset):
        plot = Plot(sample_dense_dataset)

        np.testing.assert_allclose(
            plot.dataset.da.values,
            sample_dense_dataset.to_signed_longitude().da.values,
        )
        assert plot.app is not None
        assert plot.app.name == "climatrix.plot.core"

    def test_plot_static_folder_setup(self, sample_dense_dataset):
        plot = Plot(sample_dense_dataset)

        expected_static = importlib.resources.files(
            "climatrix.resources"
        ).joinpath("static")
        assert plot.app.static_folder == str(expected_static)

    def test_routes_are_registered(self, sample_dense_dataset):
        plot = Plot(sample_dense_dataset)

        rule_endpoints = [
            rule.endpoint for rule in plot.app.url_map.iter_rules()
        ]

        assert "index" in rule_endpoints
        assert "serve_mylibrary_asset" in rule_endpoints
        assert "get_data" in rule_endpoints
        assert "get_metadata" in rule_endpoints


class TestPlotMetadata:

    def test_get_metadata_dense_with_time_and_vertical(
        self, sample_dataset_with_vertical
    ):
        plot = Plot(sample_dataset_with_vertical)
        metadata = plot.get_metadata()

        assert metadata["has_time"] is True
        assert metadata["has_vertical"] is True
        assert metadata["is_sparse"] is False
        assert metadata["time_count"] == 2
        assert metadata["vertical_count"] == 3
        assert metadata["vertical_name"] == "level"
        assert len(metadata["time_values"]) == 2
        assert len(metadata["vertical_values"]) == 3

        assert all(isinstance(tv, str) for tv in metadata["time_values"])
        assert all(
            isinstance(vv, (int, float)) for vv in metadata["vertical_values"]
        )

    def test_get_metadata_sparse_dataset(self, sample_sparse_dataset):
        plot = Plot(sample_sparse_dataset)
        metadata = plot.get_metadata()

        assert metadata["has_time"] is True
        assert metadata["has_vertical"] is False
        assert metadata["is_sparse"] is True
        assert metadata["time_count"] == 2
        assert "vertical_count" not in metadata
        assert "vertical_name" not in metadata
        assert "vertical_values" not in metadata

    def test_get_metadata_static_dataset(self, sample_static_dense_dataset):
        plot = Plot(sample_static_dense_dataset)
        metadata = plot.get_metadata()

        assert metadata["has_time"] is False
        assert metadata["has_vertical"] is False
        assert metadata["is_sparse"] is False
        assert "time_count" not in metadata
        assert "time_values" not in metadata


class TestDataPreparation:

    def test_prepare_data_calls_correct_method_for_sparse(
        self, sample_sparse_dataset
    ):
        plot = Plot(sample_sparse_dataset)

        with patch.object(plot, "prepare_sparse_data") as mock_sparse:
            mock_sparse.return_value = {"type": "scatter"}
            result = plot.prepare_data(time_idx=1, vertical_idx=0)

            mock_sparse.assert_called_once_with(1, 0)
            assert result["type"] == "scatter"

    def test_prepare_data_calls_correct_method_for_dense(
        self, sample_dense_dataset
    ):
        plot = Plot(sample_dense_dataset)

        with patch.object(plot, "prepare_dense_data") as mock_dense:
            mock_dense.return_value = {"type": "mesh"}
            result = plot.prepare_data(time_idx=1, vertical_idx=0)

            mock_dense.assert_called_once_with(1, 0, 1, None, None, None, None)
            assert result["type"] == "mesh"

    def test_prepare_sparse_data_structure(self, sample_sparse_dataset):
        plot = Plot(sample_sparse_dataset)
        result = plot.prepare_sparse_data(time_idx=0, vertical_idx=0)

        assert result["type"] == "scatter"
        assert "lats" in result
        assert "lons" in result
        assert "values" in result
        assert "min_val" in result
        assert "max_val" in result

        assert isinstance(result["lats"], list)
        assert isinstance(result["lons"], list)
        assert isinstance(result["values"], list)
        assert isinstance(result["min_val"], float)
        assert isinstance(result["max_val"], float)

        assert len(result["lats"]) == len(result["lons"])
        assert len(result["lats"]) == len(result["values"])

    def test_prepare_dense_data_structure(self, sample_dense_dataset):
        plot = Plot(sample_dense_dataset)
        result = plot.prepare_dense_data(time_idx=0, vertical_idx=0)

        assert result["type"] == "mesh"
        assert "lats" in result
        assert "lons" in result
        assert "values" in result
        assert "min_val" in result
        assert "max_val" in result

        assert isinstance(result["lats"], list)
        assert isinstance(result["lons"], list)
        assert isinstance(result["values"], list)
        assert isinstance(result["min_val"], float)
        assert isinstance(result["max_val"], float)

        assert all(isinstance(row, list) for row in result["values"])

    def test_prepare_data_with_time_indexing(self, sample_dense_dataset):
        plot = Plot(sample_dense_dataset)

        result_t0 = plot.prepare_dense_data(time_idx=0, vertical_idx=0)
        result_t1 = plot.prepare_dense_data(time_idx=1, vertical_idx=0)

        assert result_t0["type"] == result_t1["type"]
        assert len(result_t0["lats"]) == len(result_t1["lats"])
        assert len(result_t0["lons"]) == len(result_t1["lons"])

        np.testing.assert_raises(
            AssertionError,
            np.testing.assert_array_equal,
            result_t0["values"],
            result_t1["values"],
        )

    def test_prepare_data_with_vertical_indexing(
        self, sample_dataset_with_vertical
    ):
        plot = Plot(sample_dataset_with_vertical)

        result_l0 = plot.prepare_dense_data(time_idx=0, vertical_idx=0)
        result_l1 = plot.prepare_dense_data(time_idx=0, vertical_idx=1)

        assert result_l0["type"] == result_l1["type"]
        assert len(result_l0["lats"]) == len(result_l1["lats"])
        assert len(result_l0["lons"]) == len(result_l1["lons"])

        np.testing.assert_raises(
            AssertionError,
            np.testing.assert_array_equal,
            result_l0["values"],
            result_l1["values"],
        )

    def test_prepare_data_handles_static_dataset(
        self, sample_static_dense_dataset
    ):
        plot = Plot(sample_static_dense_dataset)
        result = plot.prepare_dense_data(time_idx=0, vertical_idx=0)

        assert result["type"] == "mesh"
        assert "lats" in result
        assert "lons" in result
        assert "values" in result
        assert isinstance(result["min_val"], float)
        assert isinstance(result["max_val"], float)

    def test_prepare_data_handles_edge_cases(self, sample_dense_dataset):
        plot = Plot(sample_dense_dataset)

        result = plot.prepare_dense_data(time_idx=0, vertical_idx=0)
        assert result is not None

    def test_min_max_values_are_computed_correctly(self, sample_dense_dataset):
        plot = Plot(sample_dense_dataset)
        result = plot.prepare_dense_data(time_idx=0, vertical_idx=0)

        data_slice = sample_dense_dataset.da.values
        expected_min = float(np.min(data_slice))
        expected_max = float(np.max(data_slice))

        assert result["min_val"] == expected_min
        assert result["max_val"] == expected_max


class TestPlotHTMLTemplate:

    @patch("importlib.resources.files")
    def test_get_html_template_reads_correct_file(
        self, mock_files, sample_dense_dataset
    ):
        mock_template_content = "<html>Test Template</html>"
        mock_path = MagicMock()
        mock_path.read_text.return_value = mock_template_content
        mock_files.return_value.joinpath.return_value = mock_path

        plot = Plot(sample_dense_dataset)
        mock_files.assert_called_once_with("climatrix.resources")
        result = plot.get_html_template()

        assert mock_files.call_count == 2
        assert (
            call("static", "plot_template.html")
            in mock_files.return_value.joinpath.call_args_list
        )
        mock_path.read_text.assert_called_once()
        assert result == mock_template_content

    def test_get_html_template_returns_string(self, sample_dense_dataset):
        plot = Plot(sample_dense_dataset)
        result = plot.get_html_template()

        assert isinstance(result, str)
        assert len(result) > 0


class TestPlotFlaskRoutes:

    def test_index_route_returns_html(self, sample_dense_dataset):
        plot = Plot(sample_dense_dataset)

        with plot.app.test_client() as client:
            response = client.get("/")
            assert response.status_code == 200
            assert response.content_type.startswith("text/html")

    def test_api_metadata_route_returns_json(self, sample_dense_dataset):
        plot = Plot(sample_dense_dataset)

        with plot.app.test_client() as client:
            response = client.get("/api/metadata")
            assert response.status_code == 200
            assert response.content_type == "application/json"

            data = response.get_json()
            assert isinstance(data, dict)
            assert "has_time" in data
            assert "has_vertical" in data
            assert "is_sparse" in data

    def test_api_data_route_returns_json(self, sample_dense_dataset):
        plot = Plot(sample_dense_dataset)

        with plot.app.test_client() as client:
            response = client.get("/api/data")
            assert response.status_code == 200
            assert response.content_type == "application/json"

            data = response.get_json()
            assert isinstance(data, dict)
            assert "type" in data
            assert "lats" in data
            assert "lons" in data
            assert "values" in data

    def test_api_data_route_accepts_parameters(
        self, sample_dataset_with_vertical
    ):
        plot = Plot(sample_dataset_with_vertical)

        with plot.app.test_client() as client:
            response = client.get("/api/data?time_idx=1&vertical_idx=2")
            assert response.status_code == 200

            data = response.get_json()
            assert isinstance(data, dict)
            assert "type" in data
            assert "values" in data

    def test_static_asset_route(self, sample_dense_dataset):
        plot = Plot(sample_dense_dataset)

        with plot.app.test_client() as client:
            response = client.get("/climatrix_assets/logo.svg")
            assert response.status_code in [200, 404]


class TestPlotShowMethod:

    @patch("webbrowser.open")
    @patch("threading.Thread")
    @patch("time.sleep")
    def test_show_starts_server_thread(
        self, mock_sleep, mock_thread, mock_browser, sample_dense_dataset
    ):
        plot = Plot(sample_dense_dataset)

        mock_thread_instance = MagicMock()
        mock_thread.return_value = mock_thread_instance

        mock_sleep.side_effect = [None, KeyboardInterrupt()]

        try:
            plot.show(port=5001, debug=False)
        except KeyboardInterrupt:
            pass

        mock_thread.assert_called_once()
        mock_thread_instance.start.assert_called_once()
        mock_thread_instance.daemon = True

        mock_browser.assert_called_once_with("http://localhost:5001")

    @patch("builtins.print")
    @patch("webbrowser.open")
    @patch("threading.Thread")
    @patch("time.sleep")
    def test_show_prints_status_messages(
        self,
        mock_sleep,
        mock_thread,
        mock_browser,
        mock_print,
        sample_dense_dataset,
    ):
        plot = Plot(sample_dense_dataset)

        mock_thread_instance = MagicMock()
        mock_thread.return_value = mock_thread_instance
        mock_sleep.side_effect = [None, KeyboardInterrupt()]

        try:
            plot.show(port=5001)
        except KeyboardInterrupt:
            pass

        print_calls = [call[0][0] for call in mock_print.call_args_list]
        assert any(
            "server started at http://localhost:5001" in msg
            for msg in print_calls
        )
        assert any("Press Ctrl+C to stop" in msg for msg in print_calls)


class TestPlotErrorHandling:

    def test_plot_handles_invalid_time_index_gracefully(
        self, sample_dense_dataset
    ):
        plot = Plot(sample_dense_dataset)

        try:
            result = plot.prepare_dense_data(time_idx=999, vertical_idx=0)
            assert isinstance(result, dict)
        except (IndexError, KeyError):
            pass

    def test_plot_handles_missing_vertical_dimension(
        self, sample_dense_dataset
    ):
        plot = Plot(sample_dense_dataset)

        result = plot.prepare_dense_data(time_idx=0, vertical_idx=999)
        assert isinstance(result, dict)
        assert result["type"] == "mesh"


class TestPlotIntegrationWithDataset:

    def test_plot_works_with_different_coordinate_names(self):
        latitude = np.array([-45.0, 0.0, 45.0])
        longitude = np.array([0.0, 180.0, 360.0])
        data = np.random.rand(3, 3)

        da = xr.DataArray(
            data,
            coords=[("latitude", latitude), ("longitude", longitude)],
            name="temperature",
        )
        dataset = BaseClimatrixDataset(da)
        plot = Plot(dataset)

        result = plot.prepare_dense_data(time_idx=0, vertical_idx=0)
        assert result["type"] == "mesh"
        assert len(result["lats"]) == 3
        assert len(result["lons"]) == 3

    def test_plot_preserves_data_values(self, sample_static_dense_dataset):
        plot = Plot(sample_static_dense_dataset)
        result = plot.prepare_dense_data(time_idx=0, vertical_idx=0)

        original_values = (
            sample_static_dense_dataset.to_signed_longitude().da.values
        )
        result_values = np.array(result["values"])

        np.testing.assert_array_equal(original_values, result_values)

    def test_plot_handles_single_point_sparse_data(self):
        lat = np.array([45.0])
        lon = np.array([90.0])
        data = np.array([25.5])

        da = xr.DataArray(
            data,
            coords={
                "point": [0],
                "lat": ("point", lat),
                "lon": ("point", lon),
            },
            dims=["point"],
            name="temperature",
        )
        dataset = BaseClimatrixDataset(da)
        plot = Plot(dataset)

        result = plot.prepare_sparse_data(time_idx=0, vertical_idx=0)

        assert result["type"] == "scatter"
        assert len(result["lats"]) == 1
        assert len(result["lons"]) == 1
        assert len(result["values"]) == 1
        assert result["lats"][0] == 45.0
        assert result["lons"][0] == 90.0
        assert result["values"][0] == 25.5
