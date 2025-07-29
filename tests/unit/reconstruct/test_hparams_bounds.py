"""Tests for reconstruction method hyperparameter bounds."""

import pytest

from climatrix.reconstruct.idw import IDWReconstructor
from climatrix.reconstruct.kriging import OrdinaryKrigingReconstructor


class TestReconstructorBounds:
    """Test hyperparameter bounds methods on reconstructors."""
    
    def test_idw_get_hparams_bounds(self):
        """Test IDW reconstructor bounds."""
        bounds = IDWReconstructor.get_hparams_bounds()
        
        assert "power" in bounds
        assert "k" in bounds
        assert "k_min" in bounds
        
        # Check bounds are tuples with reasonable values
        assert isinstance(bounds["power"], tuple)
        assert len(bounds["power"]) == 2
        assert bounds["power"][0] < bounds["power"][1]
        
        assert isinstance(bounds["k"], tuple)
        assert len(bounds["k"]) == 2
        assert bounds["k"][0] < bounds["k"][1]
        
        assert isinstance(bounds["k_min"], tuple)
        assert len(bounds["k_min"]) == 2
        assert bounds["k_min"][0] < bounds["k_min"][1]
        
    def test_kriging_get_hparams_bounds(self):
        """Test Kriging reconstructor bounds."""
        bounds = OrdinaryKrigingReconstructor.get_hparams_bounds()
        
        # Kriging might have empty bounds initially since parameters
        # are passed via pykrige_kwargs
        assert isinstance(bounds, dict)