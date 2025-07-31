"""Tests for the plot module."""

import pytest
import numpy as np
import xarray as xr
from unittest.mock import patch, MagicMock

from climatrix.plot.core import Plot, _check_plotting_dependencies
from climatrix.exceptions import MissingDependencyError


class TestPlotDependencies:
    """Test dependency checking for plotting."""
    
    def test_missing_plotly_dependency(self):
        """Test error when plotly is missing."""
        with patch.dict('sys.modules', {'plotly': None}):
            with pytest.raises(MissingDependencyError) as excinfo:
                _check_plotting_dependencies()
            assert "plotly and dash" in str(excinfo.value)
    
    def test_missing_dash_dependency(self):
        """Test error when dash is missing."""
        with patch.dict('sys.modules', {'dash': None}):
            with pytest.raises(MissingDependencyError) as excinfo:
                _check_plotting_dependencies()
            assert "plotly and dash" in str(excinfo.value)


class TestPlotCore:
    """Test core plotting functionality."""
    
    @pytest.fixture
    def simple_dataset(self):
        """Create a simple test dataset."""
        # Create simple 2D spatial dataset
        lats = np.linspace(-90, 90, 10)
        lons = np.linspace(-180, 180, 20)
        
        # Create temperature data
        lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
        temp_data = 15 - 0.5 * np.abs(lat_grid) + np.random.normal(0, 1, lat_grid.shape)
        
        ds = xr.Dataset(
            {'temperature': (['latitude', 'longitude'], temp_data)},
            coords={'latitude': lats, 'longitude': lons}
        )
        return ds.cm
    
    @pytest.fixture
    def temporal_dataset(self):
        """Create a dataset with time dimension."""
        lats = np.linspace(-50, 50, 5)
        lons = np.linspace(-120, 120, 10)
        times = np.arange(np.datetime64('2020-01-01'), np.datetime64('2020-01-05'), np.timedelta64(1, 'D'))
        
        # Create 3D temperature data (time, lat, lon)
        temp_data = np.random.normal(15, 5, (len(times), len(lats), len(lons)))
        
        ds = xr.Dataset(
            {'temperature': (['time', 'latitude', 'longitude'], temp_data)},
            coords={'time': times, 'latitude': lats, 'longitude': lons}
        )
        return ds.cm
    
    @pytest.fixture
    def sparse_dataset(self):
        """Create a sparse station dataset."""
        n_stations = 10
        station_lats = np.random.uniform(-90, 90, n_stations)
        station_lons = np.random.uniform(-180, 180, n_stations)
        station_ids = [f"STN_{i:03d}" for i in range(n_stations)]
        
        # Create precipitation data
        precip_data = np.random.exponential(5, n_stations)
        
        ds = xr.Dataset(
            {'precipitation': (['station'], precip_data)},
            coords={
                'station': station_ids,
                'latitude': ('station', station_lats),
                'longitude': ('station', station_lons)
            }
        )
        return ds.cm
    
    def test_plot_init_missing_dependencies(self, simple_dataset):
        """Test Plot initialization with missing dependencies."""
        with patch('climatrix.plot.core._check_plotting_dependencies') as mock_check:
            mock_check.side_effect = MissingDependencyError("Missing deps")
            
            with pytest.raises(MissingDependencyError):
                Plot(simple_dataset)
    
    @patch('climatrix.plot.core._check_plotting_dependencies')
    @patch('dash.Dash')
    def test_plot_init_success(self, mock_dash, mock_check, simple_dataset):
        """Test successful Plot initialization."""
        # Mock the Dash app
        mock_app = MagicMock()
        mock_dash.return_value = mock_app
        
        # Mock required imports
        with patch.dict('sys.modules', {
            'dash': MagicMock(),
            'plotly.graph_objects': MagicMock(),
            'plotly.express': MagicMock()
        }):
            plot = Plot(simple_dataset, auto_open=False)
            
            assert plot.dataset == simple_dataset
            assert plot.port == 8050
            assert plot.host == "127.0.0.1"
            assert not plot.auto_open
    
    @patch('climatrix.plot.core._check_plotting_dependencies')
    @patch('dash.Dash')
    def test_dataset_analysis(self, mock_dash, mock_check, temporal_dataset):
        """Test dataset analysis functionality."""
        with patch.dict('sys.modules', {
            'dash': MagicMock(),
            'plotly.graph_objects': MagicMock(),
            'plotly.express': MagicMock()
        }):
            plot = Plot(temporal_dataset, auto_open=False)
            
            # Should detect time dimension
            assert plot.has_time
            assert 'time' in plot.coords
            
            # Should have spatial coordinates
            assert 'lat' in plot.coords
            assert 'lon' in plot.coords
    
    @patch('climatrix.plot.core._check_plotting_dependencies')
    @patch('dash.Dash')
    def test_sparse_detection(self, mock_dash, mock_check, sparse_dataset):
        """Test sparse dataset detection."""
        with patch.dict('sys.modules', {
            'dash': MagicMock(),
            'plotly.graph_objects': MagicMock(),
            'plotly.express': MagicMock()
        }):
            plot = Plot(sparse_dataset, auto_open=False)
            
            # Should be detected as sparse
            assert plot.is_sparse
    
    @patch('climatrix.plot.core._check_plotting_dependencies')
    @patch('dash.Dash')
    @patch('plotly.graph_objects')
    def test_plot_generation(self, mock_go, mock_dash, mock_check, simple_dataset):
        """Test plot generation functionality."""
        # Mock plotly objects
        mock_fig = MagicMock()
        mock_go.Figure.return_value = mock_fig
        mock_go.Scattergeo = MagicMock()
        
        with patch.dict('sys.modules', {
            'dash': MagicMock(),
            'plotly.graph_objects': mock_go,
            'plotly.express': MagicMock(),
            'numpy': np
        }):
            plot = Plot(simple_dataset, auto_open=False)
            
            # Test 2D plot generation with projection
            fig = plot._generate_plot('equirectangular')
            assert fig is not None
    
    @patch('climatrix.plot.core._check_plotting_dependencies')
    @patch('dash.Dash')
    @patch('webbrowser.open')
    def test_show_method(self, mock_browser, mock_dash, mock_check, simple_dataset):
        """Test the show method."""
        mock_app = MagicMock()
        mock_dash.return_value = mock_app
        
        with patch.dict('sys.modules', {
            'dash': MagicMock(),
            'plotly.graph_objects': MagicMock(),
            'plotly.express': MagicMock()
        }):
            plot = Plot(simple_dataset, auto_open=True)
            
            # Mock the server run to avoid actually starting it
            mock_app.run_server.side_effect = KeyboardInterrupt()
            
            # Test show method
            plot.show()
            
            # Should have called run_server
            mock_app.run_server.assert_called_once()
    
    @patch('climatrix.plot.core._check_plotting_dependencies')
    @patch('dash.Dash')
    def test_save_html(self, mock_dash, mock_check, simple_dataset):
        """Test saving plot as HTML."""
        mock_fig = MagicMock()
        
        with patch.dict('sys.modules', {
            'dash': MagicMock(),
            'plotly.graph_objects': MagicMock(),
            'plotly.express': MagicMock()
        }):
            plot = Plot(simple_dataset, auto_open=False)
            
            # Mock _generate_plot to return mock figure
            plot._generate_plot = MagicMock(return_value=mock_fig)
            
            # Test save_html
            plot.save_html("/tmp/test_plot.html")
            
            # Should have called write_html on the figure
            mock_fig.write_html.assert_called_once_with("/tmp/test_plot.html")


class TestPlotIntegration:
    """Integration tests for plotting functionality."""
    
    def test_plot_import(self):
        """Test that plot module can be imported."""
        try:
            from climatrix.plot import Plot
            assert Plot is not None
        except ImportError:
            pytest.skip("Plot module not available")
    
    def test_climatrix_plot_attribute(self):
        """Test that cm.plot is available."""
        try:
            import climatrix as cm
            assert hasattr(cm, 'plot')
            assert hasattr(cm.plot, 'Plot')
        except ImportError:
            pytest.skip("Climatrix not available")