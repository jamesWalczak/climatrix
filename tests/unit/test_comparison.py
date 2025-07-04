import unittest.mock as mock

import numpy as np
import pytest
import xarray as xr

from climatrix.comparison import Comparison
from climatrix.dataset.base import BaseClimatrixDataset


@pytest.fixture
def sample_datasets():
    """Create sample datasets for testing comparison functionality."""
    # Create simple static datasets for testing
    lat = np.array([-45.0, 0.0, 45.0])
    lon = np.array([0.0, 180.0, 360.0])
    
    # Create predicted dataset
    predicted_data = np.arange(9).reshape(3, 3) + 1.0
    predicted_da = xr.DataArray(
        predicted_data,
        coords=[("lat", lat), ("lon", lon)],
        name="predicted",
    )
    predicted_dataset = BaseClimatrixDataset(predicted_da)
    
    # Create true dataset (slightly different values)
    true_data = np.arange(9).reshape(3, 3) + 0.5
    true_da = xr.DataArray(
        true_data,
        coords=[("lat", lat), ("lon", lon)],
        name="true",
    )
    true_dataset = BaseClimatrixDataset(true_da)
    
    return predicted_dataset, true_dataset


class TestComparison:
    """Test class for Comparison functionality."""
    
    def test_plot_diff_accepts_parameters(self, sample_datasets):
        """Test that plot_diff method accepts and passes through plotting parameters."""
        predicted_dataset, true_dataset = sample_datasets
        comparison = Comparison(predicted_dataset, true_dataset)
        
        # Mock the underlying plot method to verify parameters are passed through
        with mock.patch.object(comparison.diff, "plot") as mock_plot:
            # Call plot_diff with various parameters
            comparison.plot_diff(
                title="Test Title",
                show=False,
                figsize=(10, 8),
                cmap="viridis"
            )
            
            # Verify that plot was called with the expected parameters
            mock_plot.assert_called_once_with(
                title="Test Title",
                target=None,
                show=False,
                figsize=(10, 8),
                cmap="viridis"
            )
    
    def test_plot_diff_with_defaults(self, sample_datasets):
        """Test that plot_diff method works with default parameters."""
        predicted_dataset, true_dataset = sample_datasets
        comparison = Comparison(predicted_dataset, true_dataset)
        
        # Mock the underlying plot method
        with mock.patch.object(comparison.diff, "plot") as mock_plot:
            # Call plot_diff with no parameters
            comparison.plot_diff()
            
            # Verify that plot was called with default parameters
            mock_plot.assert_called_once_with(
                title=None,
                target=None,
                show=True
            )
    
    def test_plot_diff_with_ax_parameter(self, sample_datasets):
        """Test that plot_diff method accepts ax parameter through kwargs."""
        predicted_dataset, true_dataset = sample_datasets
        comparison = Comparison(predicted_dataset, true_dataset)
        
        # Mock the underlying plot method and create a mock axis
        with mock.patch.object(comparison.diff, "plot") as mock_plot:
            mock_ax = mock.Mock()
            
            # Call plot_diff with ax parameter
            comparison.plot_diff(ax=mock_ax)
            
            # Verify that plot was called with the ax parameter
            mock_plot.assert_called_once_with(
                title=None,
                target=None,
                show=True,
                ax=mock_ax
            )
