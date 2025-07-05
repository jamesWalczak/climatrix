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


@pytest.fixture
def sample_sparse_datasets():
    """Create sample sparse datasets for testing comparison functionality."""
    # Create sparse datasets
    points = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    
    predicted_values = np.array([1.0, 2.0, 3.0])
    predicted_da = xr.DataArray(
        predicted_values,
        coords=[("point", np.arange(len(predicted_values)))],
        dims=["point"],
        name="predicted",
    ).assign_coords({
        "latitude": ("point", points[:, 0]),
        "longitude": ("point", points[:, 1]),
    })
    predicted_dataset = BaseClimatrixDataset(predicted_da)
    
    true_values = np.array([1.5, 2.5, 3.5])
    true_da = xr.DataArray(
        true_values,
        coords=[("point", np.arange(len(true_values)))],
        dims=["point"],
        name="true",
    ).assign_coords({
        "latitude": ("point", points[:, 0]),
        "longitude": ("point", points[:, 1]),
    })
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

    def test_sparse_domain_comparison_basic(self, sample_sparse_datasets):
        """Test basic sparse domain comparison functionality."""
        predicted_dataset, true_dataset = sample_sparse_datasets
        comparison = Comparison(predicted_dataset, true_dataset)
        
        # Should compute differences successfully
        expected_diff = np.array([-0.5, -0.5, -0.5])
        np.testing.assert_array_almost_equal(comparison.diff.da.values, expected_diff)
        
        # Should be able to compute metrics
        rmse = comparison.compute_rmse()
        assert abs(rmse - 0.5) < 1e-10
        
    def test_sparse_domain_comparison_with_distance_threshold(self, sample_sparse_datasets):
        """Test sparse domain comparison with distance threshold."""
        predicted_dataset, true_dataset = sample_sparse_datasets
        
        # Create offset true dataset
        offset_points = np.array([[0.1, 0.1], [1.1, 1.1], [2.1, 2.1]])
        true_values = np.array([1.5, 2.5, 3.5])
        offset_true_da = xr.DataArray(
            true_values,
            coords=[("point", np.arange(len(true_values)))],
            dims=["point"],
            name="true",
        ).assign_coords({
            "latitude": ("point", offset_points[:, 0]),
            "longitude": ("point", offset_points[:, 1]),
        })
        offset_true_dataset = BaseClimatrixDataset(offset_true_da)
        
        # Test with large threshold - should match all points
        comparison_large = Comparison(predicted_dataset, offset_true_dataset, 
                                    distance_threshold=1.0)
        assert len(comparison_large.diff.da.values) == 3
        
        # Test with small threshold - should match no points
        comparison_small = Comparison(predicted_dataset, offset_true_dataset, 
                                    distance_threshold=0.1)
        assert len(comparison_small.diff.da.values) == 0
        
    def test_mixed_domain_error(self, sample_datasets, sample_sparse_datasets):
        """Test that mixing sparse and dense domains raises an error."""
        dense_dataset, _ = sample_datasets
        sparse_dataset, _ = sample_sparse_datasets
        
        with pytest.raises(ValueError, match="Comparison between sparse and dense domains is not supported"):
            Comparison(dense_dataset, sparse_dataset)
            
        with pytest.raises(ValueError, match="Comparison between sparse and dense domains is not supported"):
            Comparison(sparse_dataset, dense_dataset)
            
    def test_distance_threshold_parameter_in_init(self, sample_sparse_datasets):
        """Test that distance_threshold parameter is stored correctly."""
        predicted_dataset, true_dataset = sample_sparse_datasets
        
        # Test without distance threshold
        comparison1 = Comparison(predicted_dataset, true_dataset)
        assert comparison1.distance_threshold is None
        
        # Test with distance threshold
        comparison2 = Comparison(predicted_dataset, true_dataset, distance_threshold=0.5)
        assert comparison2.distance_threshold == 0.5
