import unittest.mock as mock

import numpy as np
import pytest
import xarray as xr

from climatrix.comparison import Comparison
from climatrix.dataset.base import BaseClimatrixDataset


@pytest.fixture
def sample_datasets():
    """Create sample datasets for testing comparison functionality."""
    lat = np.array([-45.0, 0.0, 45.0])
    lon = np.array([0.0, 180.0, 360.0])
    
    predicted_data = np.arange(9).reshape(3, 3) + 1.0
    predicted_da = xr.DataArray(
        predicted_data,
        coords=[("lat", lat), ("lon", lon)],
        name="predicted",
    )
    predicted_dataset = BaseClimatrixDataset(predicted_da)
    
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
        
        with mock.patch('climatrix.dataset.base.BaseClimatrixDataset.plot') as mock_plot:
            comparison.plot_diff(
                title="Test Title",
                show=True,
                ax=None,
                figsize=(10, 8),
                cmap="viridis"
            )
            
            mock_plot.assert_called_once_with(
                title="Test Title",
                target=None,
                show=True,
                ax=None,
                figsize=(10, 8),
                cmap="viridis"
            )
    
    def test_plot_diff_with_defaults(self, sample_datasets):
        """Test that plot_diff method works with default parameters."""
        predicted_dataset, true_dataset = sample_datasets
        comparison = Comparison(predicted_dataset, true_dataset)
        
        with mock.patch('climatrix.dataset.base.BaseClimatrixDataset.plot') as mock_plot:
            comparison.plot_diff()
            
            mock_plot.assert_called_once_with(
                title=None,
                target=None,
                show=False,
                ax=None
            )
    
    def test_plot_diff_with_ax_parameter(self, sample_datasets):
        """Test that plot_diff method accepts explicit ax parameter."""
        predicted_dataset, true_dataset = sample_datasets
        comparison = Comparison(predicted_dataset, true_dataset)
        
        with mock.patch('climatrix.dataset.base.BaseClimatrixDataset.plot') as mock_plot:
            mock_ax = mock.Mock()
            comparison.plot_diff(ax=mock_ax)
            
            mock_plot.assert_called_once_with(
                title=None,
                target=None,
                show=False,
                ax=mock_ax
            )
