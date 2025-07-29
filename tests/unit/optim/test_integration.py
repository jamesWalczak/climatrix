"""Integration tests for hyperparameter optimization without requiring bayesian-optimization."""

from datetime import datetime
import numpy as np
import pytest
import xarray as xr

from climatrix import BaseClimatrixDataset


class TestHParamFinderIntegration:
    """Integration tests that can run without bayesian-optimization package."""
    
    @pytest.fixture
    def train_dataset(self):
        """Create a simple training dataset."""
        return BaseClimatrixDataset(
            xr.DataArray(
                data=np.random.rand(8, 1),
                dims=("point", "time"),
                coords={
                    "point": np.arange(8),
                    "time": np.array([datetime(2000, 1, 1)], dtype="datetime64"),
                    "latitude": (("point",), np.linspace(-70, 70, 8)),
                    "longitude": (("point",), np.linspace(-140, 140, 8)),
                },
            )
        )
    
    @pytest.fixture
    def val_dataset(self):
        """Create a simple validation dataset."""
        return BaseClimatrixDataset(
            xr.DataArray(
                data=np.random.rand(4, 1),
                dims=("point", "time"),
                coords={
                    "point": np.arange(4),
                    "time": np.array([datetime(2000, 1, 1)], dtype="datetime64"),
                    "latitude": (("point",), np.linspace(-50, 50, 4)),
                    "longitude": (("point",), np.linspace(-100, 100, 4)),
                },
            )
        )

    def test_interface_matches_specification(self, train_dataset, val_dataset):
        """Test that the interface exactly matches the specification."""
        try:
            from climatrix.optim.bayesian import HParamFinder
            
            # This is the exact interface from the issue specification
            finder = HParamFinder(
                train_dataset, 
                val_dataset, 
                metric="mae", 
                method="idw", 
                exclude="k", 
                include=["power", "k_min"], 
                explore=0.9, 
                n_iters=100
            )
            
            # Verify the object was created successfully
            assert finder is not None
            assert finder.metric == "mae"
            assert finder.method == "idw"
            assert finder.explore == 0.9
            assert finder.n_iters == 100
            
            # Verify filtering worked
            assert "power" in finder.bounds
            assert "k_min" in finder.bounds
            assert "k" not in finder.bounds
            
            # Verify init_points and n_iter calculation
            assert finder.init_points == 90  # 0.9 * 100
            assert finder.n_iter == 10      # 100 - 90
            
        except ImportError:
            pytest.skip("Requires climatrix package to be properly installed")
            
    def test_get_hparams_bounds_function(self):
        """Test the standalone get_hparams_bounds function."""
        try:
            from climatrix.optim.bayesian import get_hparams_bounds
            
            # Test IDW bounds
            idw_bounds = get_hparams_bounds("idw")
            assert isinstance(idw_bounds, dict)
            assert "power" in idw_bounds
            assert "k" in idw_bounds
            assert "k_min" in idw_bounds
            
            # Test kriging bounds
            kriging_bounds = get_hparams_bounds("kriging")
            assert isinstance(kriging_bounds, dict)
            
            # Test unknown method
            with pytest.raises(ValueError, match="Unknown reconstruction method"):
                get_hparams_bounds("nonexistent_method")
                
        except ImportError:
            pytest.skip("Requires climatrix package to be properly installed")
            
    def test_parameter_processing_and_validation(self, train_dataset, val_dataset):
        """Test parameter processing without running optimization."""
        try:
            from climatrix.optim.bayesian import HParamFinder
            
            finder = HParamFinder(train_dataset, val_dataset)
            
            # Test parameter processing
            params = {
                "power": 2.3,
                "k": 7.8,
                "k_min": 2.1
            }
            
            processed = finder._process_params(params)
            
            assert processed["power"] == 2.3  # Float remains float
            assert processed["k"] == 8        # Float converted to int
            assert processed["k_min"] == 2    # Float converted to int
            
        except ImportError:
            pytest.skip("Requires climatrix package to be properly installed")
            
    def test_metric_calculation_functionality(self, train_dataset, val_dataset):
        """Test metric calculation logic."""
        try:
            from climatrix.optim.bayesian import HParamFinder
            
            finder = HParamFinder(train_dataset, val_dataset, metric="mae")
            
            # Create simple mock datasets for testing
            class MockDataset:
                def __init__(self, values):
                    self.da = MockDataArray(values)
                    
            class MockDataArray:
                def __init__(self, values):
                    self.values = MockValues(values)
                    
            class MockValues:
                def __init__(self, values):
                    self._values = values
                    
                def flatten(self):
                    return self._values
            
            # Test MAE calculation
            mock_recon = MockDataset(np.array([1.0, 2.0, 3.0]))
            mock_val = MockDataset(np.array([1.5, 2.5, 2.5]))
            
            mae = finder._calculate_metric(mock_recon, mock_val, "mae")
            expected_mae = np.mean([0.5, 0.5, 0.5])  # |1-1.5|, |2-2.5|, |3-2.5|
            
            assert abs(mae - expected_mae) < 1e-6
            
            # Test with NaN values
            mock_recon_nan = MockDataset(np.array([1.0, np.nan, 3.0]))
            mock_val_nan = MockDataset(np.array([1.5, 2.5, np.nan]))
            
            mae_nan = finder._calculate_metric(mock_recon_nan, mock_val_nan, "mae")
            assert abs(mae_nan - 0.5) < 1e-6  # Only |1-1.5| is valid
            
        except ImportError:
            pytest.skip("Requires climatrix package to be properly installed")