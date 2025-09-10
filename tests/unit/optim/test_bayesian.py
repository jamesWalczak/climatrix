"""Tests for Bayesian hyperparameter optimization."""

import importlib
from datetime import datetime
from unittest.mock import MagicMock

import numpy as np
import pytest
import xarray as xr

from climatrix import BaseClimatrixDataset
from climatrix.optim.bayesian import HParamFinder, MetricType
from climatrix.reconstruct.base import BaseReconstructor
from tests.unit.test_utils import skip_on_error


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
    """Test the hparams classmethod system."""

    def test_idw_hparams(self):
        """Test IDW hyperparameters."""
        from climatrix.reconstruct.idw import IDWReconstructor

        hparams = IDWReconstructor.get_hparams()

        assert "power" in hparams
        assert "k" in hparams
        assert "k_min" in hparams

        assert issubclass(hparams["power"]["type"], float)
        assert issubclass(hparams["k"]["type"], int)
        # NOTE: power is unbounded
        assert "bounds" not in hparams["power"]


class TestHParamFinder:
    """Test the HParamFinder class."""

    @pytest.fixture(autouse=True)
    def setup(self):
        yield
        for reconstructor_method in BaseReconstructor.get_available_methods():
            reconstructor_class = BaseReconstructor.get(reconstructor_method)
            for hparam in reconstructor_class.get_hparams().keys():
                getattr(reconstructor_class, hparam).restore_default_bounds()

    @pytest.fixture
    def sparse_dataset(self):
        """Create a sparse dataset for testing."""
        return BaseClimatrixDataset(
            xr.DataArray(
                data=np.random.rand(5, 1),
                dims=("point", "time"),
                coords={
                    "point": np.arange(5),
                    "time": np.array(
                        [datetime(2000, 1, 1)], dtype="datetime64"
                    ),
                    "latitude": (("point",), np.array([-90, -45, 0, 45, 90])),
                    "longitude": (
                        ("point",),
                        np.array([-180, -90, 0, 90, 180]),
                    ),
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
                    "time": np.array(
                        [datetime(2000, 1, 1)], dtype="datetime64"
                    ),
                    "latitude": (("latitude",), np.array([-45, 0, 45])),
                    "longitude": (("longitude",), np.array([-90, 0, 90])),
                },
            )
        )

    def test_init_basic(self, sparse_dataset, dense_dataset):
        """Test basic initialization."""
        finder = HParamFinder("idw", sparse_dataset, dense_dataset)

        assert finder.train_dset is sparse_dataset
        assert finder.val_dset is dense_dataset
        assert finder.metric == MetricType.MAE
        assert finder.method == "idw"
        assert finder.random_seed == 42
        assert finder.bounds is not None

    def test_init_with_parameters(self, sparse_dataset, dense_dataset):
        """Test initialization with custom parameters."""
        finder = HParamFinder(
            "ok",
            sparse_dataset,
            dense_dataset,
            metric="mse",
            n_iters=50,
            random_seed=123,
        )

        assert finder.metric == MetricType.MSE
        assert finder.method == "ok"
        assert finder.random_seed == 123
        assert finder.n_iters == 50

    def test_include_parameters(self, sparse_dataset, dense_dataset):
        """Test parameter inclusion."""
        finder = HParamFinder(
            "idw", sparse_dataset, dense_dataset, include=["power", "k"]
        )

        assert set(finder.bounds.keys()) == {"power", "k"}

    def test_exclude_parameters(self, sparse_dataset, dense_dataset):
        """Test parameter exclusion."""
        finder = HParamFinder(
            "idw", sparse_dataset, dense_dataset, exclude="k"
        )

        expected_params = {"power", "k_min"}
        assert set(finder.bounds.keys()) == expected_params

    def test_include_exclude_both(self, sparse_dataset, dense_dataset):
        """Test that include and exclude can be used together if no common keys."""
        finder = HParamFinder(
            "idw",
            sparse_dataset,
            dense_dataset,
            include=["power", "k"],
            exclude=["k_min"],
        )

        assert set(finder.bounds.keys()) == {"power", "k"}

    def test_include_exclude_common_keys(self, sparse_dataset, dense_dataset):
        """Test that include and exclude cannot have common keys."""
        with pytest.raises(
            ValueError,
            match="Cannot specify same parameters in both include and exclude",
        ):
            HParamFinder(
                "idw",
                sparse_dataset,
                dense_dataset,
                include=["power", "k"],
                exclude=["k"],
            )

    def test_custom_bounds_override(self, sparse_dataset, dense_dataset):
        """Test custom bounds override."""
        custom_bounds = {"power": (100.0, 330.0), "k": (20, 89)}
        finder = HParamFinder(
            "idw", sparse_dataset, dense_dataset, bounds=custom_bounds
        )
        assert finder.bounds["power"] == (100.0, 330.0, float)
        assert finder.bounds["k"] == (20, 89, int)

    def test_invalid_n_iters(self, sparse_dataset, dense_dataset):
        """Test invalid n_iters parameter."""
        with pytest.raises(ValueError, match="n_iters must be >= 1"):
            HParamFinder("idw", sparse_dataset, dense_dataset, n_iters=0)

    def test_invalid_metric(self, sparse_dataset, dense_dataset):
        """Test invalid metric parameter."""
        with pytest.raises(
            ValueError, match="'invalid_metric' is not a valid MetricType"
        ):
            HParamFinder(
                "idw", sparse_dataset, dense_dataset, metric="invalid_metric"
            )

    def test_invalid_dataset_types(self, sparse_dataset):
        """Test invalid dataset types."""
        with pytest.raises(
            TypeError, match="train_dset must be a BaseClimatrixDataset"
        ):
            HParamFinder("idw", "not_a_dataset", sparse_dataset)

        with pytest.raises(
            TypeError, match="val_dset must be a BaseClimatrixDataset"
        ):
            HParamFinder("idw", sparse_dataset, "not_a_dataset")

    @pytest.mark.skipif(
        not importlib.util.find_spec("optuna"),
        reason="`optuna` package is not installed",
    )
    def test_evaluate_params(self, sparse_dataset, dense_dataset):
        """Test parameter evaluation (without full optimization)."""
        from optuna import Trial

        finder = HParamFinder(
            "idw",
            sparse_dataset,
            dense_dataset,
        )
        trial = MagicMock(spec=Trial)
        trial.suggest_int.side_effect = [2, 5]
        trial.suggest_float.side_effect = [5.0]

        result = finder._evaluate_params(trial)
        assert isinstance(result, float)
        assert result >= 0

    @skip_on_error(ImportError)
    def test_optimize_mock(self, sparse_dataset, dense_dataset):
        _ = HParamFinder(
            "idw", sparse_dataset, dense_dataset, n_iters=5, include=["power"]
        ).optimize()

    def test_optimize_on_explor_high(self, sparse_dataset, dense_dataset):
        finder = HParamFinder(
            "idw",
            sparse_dataset,
            dense_dataset,
            n_iters=3,
        )
        finder.optimize()
