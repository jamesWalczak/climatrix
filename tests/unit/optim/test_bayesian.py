"""Tests for Bayesian hyperparameter optimization."""

from datetime import datetime
from unittest.mock import Mock, patch

import numpy as np
import pytest
import xarray as xr

from climatrix import BaseClimatrixDataset
from tests.unit.utils import skip_on_error


class TestHParamFinder:
    """Test HParamFinder class."""
    
    @pytest.fixture
    def train_dataset(self):
        """Create a simple training dataset."""
        return BaseClimatrixDataset(
            xr.DataArray(
                data=np.random.rand(5, 1),
                dims=("point", "time"),
                coords={
                    "point": np.arange(5),
                    "time": np.array([datetime(2000, 1, 1)], dtype="datetime64"),
                    "latitude": (("point",), np.array([-90, -45, 0, 45, 90])),
                    "longitude": (("point",), np.array([-180, -90, 0, 90, 180])),
                },
            )
        )
    
    @pytest.fixture
    def val_dataset(self):
        """Create a simple validation dataset."""
        return BaseClimatrixDataset(
            xr.DataArray(
                data=np.random.rand(3, 1),
                dims=("point", "time"),
                coords={
                    "point": np.arange(3),
                    "time": np.array([datetime(2000, 1, 1)], dtype="datetime64"),
                    "latitude": (("point",), np.array([-60, 0, 60])),
                    "longitude": (("point",), np.array([-120, 0, 120])),
                },
            )
        )
    
    def test_init_default_params(self, train_dataset, val_dataset):
        """Test HParamFinder initialization with default parameters."""
        from climatrix.optim.bayesian import HParamFinder
        
        finder = HParamFinder(train_dataset, val_dataset)
        
        assert finder.train_dset is train_dataset
        assert finder.val_dset is val_dataset
        assert finder.metric == "mae"
        assert finder.method == "idw"
        assert finder.explore == 0.9
        assert finder.n_iters == 100
        
        # Check calculated values
        assert finder.init_points == 90  # 0.9 * 100
        assert finder.n_iter == 10  # 100 - 90
        
    def test_init_custom_params(self, train_dataset, val_dataset):
        """Test HParamFinder initialization with custom parameters."""
        from climatrix.optim.bayesian import HParamFinder
        
        finder = HParamFinder(
            train_dataset, 
            val_dataset,
            metric="rmse",
            method="idw",
            explore=0.5,
            n_iters=50,
            exclude=["k"],
            include=["power", "k_min"]
        )
        
        assert finder.metric == "rmse"
        assert finder.explore == 0.5
        assert finder.n_iters == 50
        assert finder.init_points == 25  # 0.5 * 50
        assert finder.n_iter == 25  # 50 - 25
        
        # Should have filtered bounds
        assert "k" not in finder.bounds
        assert "power" in finder.bounds
        assert "k_min" in finder.bounds
        
    def test_init_invalid_explore(self, train_dataset, val_dataset):
        """Test HParamFinder with invalid explore parameter."""
        from climatrix.optim.bayesian import HParamFinder
        
        with pytest.raises(ValueError, match="explore must be between 0 and 1"):
            HParamFinder(train_dataset, val_dataset, explore=1.5)
            
        with pytest.raises(ValueError, match="explore must be between 0 and 1"):
            HParamFinder(train_dataset, val_dataset, explore=0.0)
            
    def test_init_invalid_n_iters(self, train_dataset, val_dataset):
        """Test HParamFinder with invalid n_iters parameter."""
        from climatrix.optim.bayesian import HParamFinder
        
        with pytest.raises(ValueError, match="n_iters must be >= 1"):
            HParamFinder(train_dataset, val_dataset, n_iters=0)
            
    def test_bounds_filtering_include_only(self, train_dataset, val_dataset):
        """Test bounds filtering with include parameter."""
        from climatrix.optim.bayesian import HParamFinder
        
        finder = HParamFinder(
            train_dataset, 
            val_dataset,
            include="power"
        )
        
        assert list(finder.bounds.keys()) == ["power"]
        
    def test_bounds_filtering_exclude_only(self, train_dataset, val_dataset):
        """Test bounds filtering with exclude parameter."""
        from climatrix.optim.bayesian import HParamFinder
        
        finder = HParamFinder(
            train_dataset, 
            val_dataset,
            exclude=["k", "k_min"]
        )
        
        assert "k" not in finder.bounds
        assert "k_min" not in finder.bounds
        assert "power" in finder.bounds
        
    def test_bounds_filtering_include_list(self, train_dataset, val_dataset):
        """Test bounds filtering with include as list."""
        from climatrix.optim.bayesian import HParamFinder
        
        finder = HParamFinder(
            train_dataset, 
            val_dataset,
            include=["power", "k"]
        )
        
        assert set(finder.bounds.keys()) == {"power", "k"}
        
    def test_bounds_filtering_exclude_list(self, train_dataset, val_dataset):
        """Test bounds filtering with exclude as list."""
        from climatrix.optim.bayesian import HParamFinder
        
        finder = HParamFinder(
            train_dataset, 
            val_dataset,
            exclude=["power", "k"]
        )
        
        assert "power" not in finder.bounds
        assert "k" not in finder.bounds
        assert "k_min" in finder.bounds
        
    def test_bounds_filtering_no_params_left(self, train_dataset, val_dataset):
        """Test bounds filtering that leaves no parameters."""
        from climatrix.optim.bayesian import HParamFinder
        
        with pytest.raises(ValueError, match="No hyperparameters to optimize"):
            HParamFinder(
                train_dataset, 
                val_dataset,
                include=["nonexistent_param"]
            )
            
    def test_custom_bounds_override(self, train_dataset, val_dataset):
        """Test custom bounds override default bounds."""
        from climatrix.optim.bayesian import HParamFinder
        
        finder = HParamFinder(
            train_dataset, 
            val_dataset,
            power=(1.0, 3.0)  # Custom bounds for power parameter
        )
        
        assert finder.bounds["power"] == (1.0, 3.0)
        
    def test_process_params(self, train_dataset, val_dataset):
        """Test parameter processing for type conversion."""
        from climatrix.optim.bayesian import HParamFinder
        
        finder = HParamFinder(train_dataset, val_dataset)
        
        params = {
            "power": 2.5,
            "k": 7.8,
            "k_min": 3.2
        }
        
        processed = finder._process_params(params)
        
        assert processed["power"] == 2.5  # Float remains float
        assert processed["k"] == 8  # Float converted to int
        assert processed["k_min"] == 3  # Float converted to int
        
    def test_get_reconstructor_class_idw(self, train_dataset, val_dataset):
        """Test getting IDW reconstructor class."""
        from climatrix.optim.bayesian import HParamFinder
        from climatrix.reconstruct.idw import IDWReconstructor
        
        finder = HParamFinder(train_dataset, val_dataset, method="idw")
        
        cls = finder._get_reconstructor_class("idw")
        assert cls is IDWReconstructor
        
    def test_get_reconstructor_class_kriging(self, train_dataset, val_dataset):
        """Test getting Kriging reconstructor class."""
        from climatrix.optim.bayesian import HParamFinder
        from climatrix.reconstruct.kriging import OrdinaryKrigingReconstructor
        
        finder = HParamFinder(train_dataset, val_dataset, method="kriging")
        
        cls = finder._get_reconstructor_class("kriging")
        assert cls is OrdinaryKrigingReconstructor
        
    def test_get_reconstructor_class_unknown(self, train_dataset, val_dataset):
        """Test getting unknown reconstructor class raises error."""
        from climatrix.optim.bayesian import HParamFinder
        
        finder = HParamFinder(train_dataset, val_dataset)
        
        with pytest.raises(ValueError, match="Unknown reconstruction method"):
            finder._get_reconstructor_class("unknown_method")
            
    def test_calculate_metric_mae(self, train_dataset, val_dataset):
        """Test MAE metric calculation."""
        from climatrix.optim.bayesian import HParamFinder
        
        finder = HParamFinder(train_dataset, val_dataset, metric="mae")
        
        # Create mock datasets with known values
        reconstructed = Mock()
        reconstructed.da.values.flatten = Mock(return_value=np.array([1.0, 2.0, 3.0]))
        
        validation = Mock()
        validation.da.values.flatten = Mock(return_value=np.array([1.5, 2.5, 2.5]))
        
        mae = finder._calculate_metric(reconstructed, validation, "mae")
        expected_mae = np.mean([0.5, 0.5, 0.5])  # |1-1.5|, |2-2.5|, |3-2.5|
        
        assert abs(mae - expected_mae) < 1e-6
        
    def test_calculate_metric_with_nan(self, train_dataset, val_dataset):
        """Test metric calculation with NaN values."""
        from climatrix.optim.bayesian import HParamFinder
        
        finder = HParamFinder(train_dataset, val_dataset, metric="mae")
        
        # Create mock datasets with NaN values
        reconstructed = Mock()
        reconstructed.da.values.flatten = Mock(return_value=np.array([1.0, np.nan, 3.0]))
        
        validation = Mock()
        validation.da.values.flatten = Mock(return_value=np.array([1.5, 2.5, np.nan]))
        
        mae = finder._calculate_metric(reconstructed, validation, "mae")
        expected_mae = 0.5  # Only |1-1.5| is valid
        
        assert abs(mae - expected_mae) < 1e-6
        
    def test_calculate_metric_unknown(self, train_dataset, val_dataset):
        """Test unknown metric raises error."""
        from climatrix.optim.bayesian import HParamFinder
        
        finder = HParamFinder(train_dataset, val_dataset)
        
        reconstructed = Mock()
        reconstructed.da.values.flatten = Mock(return_value=np.array([1.0, 2.0]))
        
        validation = Mock()
        validation.da.values.flatten = Mock(return_value=np.array([1.5, 2.5]))
        
        with pytest.raises(ValueError, match="Unknown metric"):
            finder._calculate_metric(reconstructed, validation, "unknown_metric")
            
    @skip_on_error(ImportError)
    def test_optimize_mock(self, train_dataset, val_dataset):
        """Test optimization with mocked bayesian optimization."""
        from climatrix.optim.bayesian import HParamFinder
        
        # Mock the BayesianOptimization to avoid actually running optimization
        with patch('climatrix.optim.bayesian.BayesianOptimization') as mock_bo:
            mock_optimizer = Mock()
            mock_optimizer.max = {
                'params': {'power': 2.5, 'k': 8, 'k_min': 3},
                'target': -0.1  # Negative because MAE is minimized
            }
            mock_bo.return_value = mock_optimizer
            
            finder = HParamFinder(train_dataset, val_dataset, n_iters=5)
            result = finder.optimize()
            
            assert 'best_params' in result
            assert 'best_score' in result
            assert 'optimizer' in result
            assert result['best_params']['power'] == 2.5
            assert result['best_score'] == -0.1
            
            # Verify optimizer was called correctly
            mock_bo.assert_called_once()
            mock_optimizer.maximize.assert_called_once_with(
                init_points=finder.init_points,
                n_iter=finder.n_iter
            )