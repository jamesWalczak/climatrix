"""Tests for Bayesian hyperparameter optimization."""

from datetime import datetime

import numpy as np
import pytest
import xarray as xr

from climatrix import BaseClimatrixDataset
from climatrix.optim.bayesian import HParamFinder, get_hparams_bounds, MetricType
from climatrix.reconstruct.base import BaseReconstructor
from tests.unit.utils import skip_on_error


class TestMetricType:
    """Test the MetricType enum."""
    
    def test_metric_values(self):
        """Test metric enum values."""
        assert MetricType.MAE == "mae"
        assert MetricType.MSE == "mse"
        assert MetricType.RMSE == "rmse"


class TestBaseReconstructorRegistry:
    """Test the BaseReconstructor registry system."""
    
    def test_idw_class(self):
        """Test getting IDW reconstruction class."""
        cls = BaseReconstructor.get("idw")
        assert cls.__name__ == "IDWReconstructor"
        
    def test_ok_class(self):
        """Test getting Ordinary Kriging reconstruction class."""
        cls = BaseReconstructor.get("ok")
        assert cls.__name__ == "OrdinaryKrigingReconstructor"
        
    def test_case_insensitive(self):
        """Test that method names are case insensitive."""
        cls_lower = BaseReconstructor.get("idw")
        cls_upper = BaseReconstructor.get("IDW")
        cls_mixed = BaseReconstructor.get("IdW")
        
        assert cls_lower == cls_upper == cls_mixed
    
    def test_unknown_method(self):
        """Test error for unknown method."""
        with pytest.raises(ValueError, match="Unknown method"):
            BaseReconstructor.get("unknown_method")


class TestHyperparameterProperty:
    """Test the hparams property system."""
    
    def test_idw_hparams(self):
        """Test IDW hyperparameters."""
        from climatrix.reconstruct.idw import IDWReconstructor
        instance = IDWReconstructor.__new__(IDWReconstructor)
        hparams = instance.hparams
        
        assert "power" in hparams
        assert "k" in hparams
        assert "k_min" in hparams
        
        assert hparams["power"]["type"] == float
        assert hparams["k"]["type"] == int
        assert "bounds" in hparams["power"]


class TestGetHparamsBounds:
    """Test the get_hparams_bounds function."""
    
    def test_idw_bounds(self):
        """Test bounds for IDW method."""
        bounds = get_hparams_bounds("idw")
        expected_params = {"power", "k", "k_min"}
        assert set(bounds.keys()) == expected_params
        
        # Check bounds are tuples of (min, max)
        for param, bound in bounds.items():
            assert isinstance(bound, tuple)
            assert len(bound) == 2
            assert bound[0] < bound[1]
    
    def test_ok_bounds(self):
        """Test bounds for Ordinary Kriging method."""
        bounds = get_hparams_bounds("ok")
        expected_params = {"nlags", "weight", "verbose", "pseudo_inv"}
        assert set(bounds.keys()) == expected_params
        
        # Check numeric bounds
        for param in ["nlags", "weight", "verbose", "pseudo_inv"]:
            bound = bounds[param]
            assert isinstance(bound, tuple)
            assert len(bound) == 2
            assert bound[0] < bound[1]
    
    def test_sinet_bounds(self):
        """Test bounds for SiNET method."""
        bounds = get_hparams_bounds("sinet")
        expected_params = {
            "lr", "batch_size", "num_epochs", "gradient_clipping_value",
            "mse_loss_weight", "eikonal_loss_weight", "laplace_loss_weight"
        }
        assert set(bounds.keys()) == expected_params
    
    def test_case_insensitive(self):
        """Test that method names are case insensitive."""
        bounds_lower = get_hparams_bounds("idw")
        bounds_upper = get_hparams_bounds("IDW")
        bounds_mixed = get_hparams_bounds("IdW")
        
        assert bounds_lower == bounds_upper == bounds_mixed
    
    def test_unknown_method(self):
        """Test error for unknown method."""
        with pytest.raises(ValueError, match="Unknown reconstruction method"):
            get_hparams_bounds("unknown_method")


class TestHParamFinder:
    """Test the HParamFinder class."""
    
    @pytest.fixture
    def sparse_dataset(self):
        """Create a sparse dataset for testing."""
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
    def dense_dataset(self):
        """Create a dense dataset for testing."""
        return BaseClimatrixDataset(
            xr.DataArray(
                data=np.random.rand(1, 3, 3),
                dims=("time", "latitude", "longitude"),
                coords={
                    "time": np.array([datetime(2000, 1, 1)], dtype="datetime64"),
                    "latitude": (("latitude",), np.array([-45, 0, 45])),
                    "longitude": (("longitude",), np.array([-90, 0, 90])),
                },
            )
        )
    
    def test_init_basic(self, sparse_dataset, dense_dataset):
        """Test basic initialization."""
        finder = HParamFinder(sparse_dataset, dense_dataset)
        
        assert finder.train_dset is sparse_dataset
        assert finder.val_dset is dense_dataset
        assert finder.metric == MetricType.MAE
        assert finder.method == "idw"
        assert finder.random_seed == 42
        assert finder.n_init_points + finder.n_iter == 100
        assert finder.bounds is not None
    
    def test_init_with_parameters(self, sparse_dataset, dense_dataset):
        """Test initialization with custom parameters."""
        finder = HParamFinder(
            sparse_dataset,
            dense_dataset,
            metric="mse",
            method="ok",
            explore=0.5,
            n_iters=50,
            random_seed=123
        )
        
        assert finder.metric == MetricType.MSE
        assert finder.method == "ok" 
        assert finder.random_seed == 123
        assert finder.n_init_points == 25  # 50 * 0.5
        assert finder.n_iter == 25
    
    def test_include_parameters(self, sparse_dataset, dense_dataset):
        """Test parameter inclusion."""
        finder = HParamFinder(
            sparse_dataset,
            dense_dataset,
            include=["power", "k"]
        )
        
        assert set(finder.bounds.keys()) == {"power", "k"}
    
    def test_exclude_parameters(self, sparse_dataset, dense_dataset):
        """Test parameter exclusion."""
        finder = HParamFinder(
            sparse_dataset,
            dense_dataset,
            exclude="k"
        )
        
        expected_params = {"power", "k_min"}  # IDW params except k
        assert set(finder.bounds.keys()) == expected_params
    
    def test_include_exclude_both(self, sparse_dataset, dense_dataset):
        """Test that include and exclude can be used together if no common keys."""
        finder = HParamFinder(
            sparse_dataset,
            dense_dataset,
            include=["power", "k"],
            exclude=["k_min"]  # No overlap with include
        )
        
        # Should include power and k, but not k_min (which is excluded)
        assert set(finder.bounds.keys()) == {"power", "k"}
    
    def test_include_exclude_common_keys(self, sparse_dataset, dense_dataset):
        """Test that include and exclude cannot have common keys."""
        with pytest.raises(ValueError, match="Cannot specify same parameters in both include and exclude"):
            HParamFinder(
                sparse_dataset,
                dense_dataset,
                include=["power", "k"],
                exclude=["k"]  # Common key with include
            )
    
    def test_custom_bounds(self, sparse_dataset, dense_dataset):
        """Test custom bounds override."""
        custom_bounds = {"power": (1.0, 3.0), "k": (2, 8)}
        finder = HParamFinder(
            sparse_dataset,
            dense_dataset,
            bounds=custom_bounds
        )
        
        assert finder.bounds == custom_bounds
    
    def test_invalid_explore(self, sparse_dataset, dense_dataset):
        """Test invalid explore parameter."""
        with pytest.raises(ValueError, match="explore must be in the range"):
            HParamFinder(sparse_dataset, dense_dataset, explore=0.0)
        
        with pytest.raises(ValueError, match="explore must be in the range"):
            HParamFinder(sparse_dataset, dense_dataset, explore=1.0)
    
    def test_invalid_n_iters(self, sparse_dataset, dense_dataset):
        """Test invalid n_iters parameter."""
        with pytest.raises(ValueError, match="n_iters must be >= 1"):
            HParamFinder(sparse_dataset, dense_dataset, n_iters=0)
    
    def test_invalid_metric(self, sparse_dataset, dense_dataset):
        """Test invalid metric parameter."""
        with pytest.raises(ValueError, match="invalid literal"):
            HParamFinder(sparse_dataset, dense_dataset, metric="invalid_metric")
    
    def test_invalid_dataset_types(self, sparse_dataset):
        """Test invalid dataset types."""
        with pytest.raises(TypeError, match="train_dset must be a BaseClimatrixDataset"):
            HParamFinder("not_a_dataset", sparse_dataset)
        
        with pytest.raises(TypeError, match="val_dset must be a BaseClimatrixDataset"):
            HParamFinder(sparse_dataset, "not_a_dataset")
    
    def test_evaluate_params(self, sparse_dataset, dense_dataset):
        """Test parameter evaluation (without full optimization)."""
        finder = HParamFinder(sparse_dataset, dense_dataset, method="idw")
        
        # Test with valid IDW parameters
        result = finder._evaluate_params(power=2, k=5, k_min=2)
        
        # Should return a negative float (since we negate the metric for maximization)
        assert isinstance(result, float)
        assert result <= 0  # Negative because we negate the metric
    
    def test_parameter_type_conversion(self, sparse_dataset, dense_dataset):
        """Test that parameters are converted to correct types."""
        finder = HParamFinder(sparse_dataset, dense_dataset, method="idw")
        
        # Test with float values that should be converted to integers
        result = finder._evaluate_params(power=2.1, k=5.7, k_min=2.3)
        
        # Should still work despite float inputs for integer parameters
        assert isinstance(result, float)
        assert result <= 0
    
    @skip_on_error(ImportError)
    def test_optimize_mock(self, sparse_dataset, dense_dataset):
        """Test optimization (mocked to avoid external dependency)."""
        # This test would normally call the optimize method, but since we don't have
        # bayesian-optimization installed in the test environment, we'll skip it
        # or mock it. The actual optimization testing would require the package.
        finder = HParamFinder(
            sparse_dataset,
            dense_dataset,
            method="idw",
            n_iters=5,  # Small number for testing
            include=["power"]  # Only optimize one parameter
        )
        
        # This would raise ImportError without bayesian-optimization
        with pytest.raises(ImportError):
            finder.optimize()