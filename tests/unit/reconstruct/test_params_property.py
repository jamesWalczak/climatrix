"""Test the num_params property for neural network reconstructors."""

import pytest
from unittest.mock import MagicMock

from climatrix.reconstruct.base import BaseReconstructor
from tests.unit.reconstruct.test_base_interface import TestBaseReconstructor


class MockNeuralReconstructor(BaseReconstructor):
    """Mock neural network reconstructor for testing."""
    
    NAME = "mock_nn"
    
    def __init__(self, dataset, target_domain, num_parameters=1000):
        super().__init__(dataset, target_domain)
        self.num_parameters = num_parameters  # For testing
    
    def init_model(self):
        """Mock init_model that returns a model-like object."""
        mock_model = MagicMock()
        
        # Create mock parameters
        mock_params = []
        # Distribute parameters evenly across 5 mock parameters
        params_per_param = self.num_parameters // 5
        for i in range(5):
            param = MagicMock()
            param.numel.return_value = params_per_param
            param.requires_grad = True
            mock_params.append(param)
        
        mock_model.parameters.return_value = mock_params
        return mock_model
    
    def reconstruct(self):
        """Mock reconstruct method."""
        return self.dataset


class MockNonNeuralReconstructor(BaseReconstructor):
    """Mock non-neural network reconstructor for testing."""
    
    NAME = "mock_traditional"
    
    def reconstruct(self):
        """Mock reconstruct method."""
        return self.dataset


class TestNumParamsProperty(TestBaseReconstructor):
    """Test the num_params property."""
    
    def test_num_params_for_neural_reconstructor(self):
        """Test that neural network reconstructors return correct parameter count."""
        dataset = self.create_sparse_static_dataset()
        reconstructor = MockNeuralReconstructor(dataset, dataset.domain, num_parameters=1000)
        
        # Should return the total number of parameters (1000)
        assert reconstructor.num_params == 1000
    
    def test_num_params_for_non_neural_reconstructor(self):
        """Test that non-neural network reconstructors return 0."""
        dataset = self.create_sparse_static_dataset()
        reconstructor = MockNonNeuralReconstructor(dataset, dataset.domain)
        
        # Should return 0 since it doesn't have init_model
        assert reconstructor.num_params == 0
    
    def test_num_params_with_non_trainable_parameters(self):
        """Test parameter counting excludes non-trainable parameters."""
        dataset = self.create_sparse_static_dataset()
        reconstructor = MockNeuralReconstructor(dataset, dataset.domain)
        
        # Mock model with mix of trainable and non-trainable parameters
        mock_model = MagicMock()
        mock_params = []
        
        # 3 trainable parameters
        for i in range(3):
            param = MagicMock()
            param.numel.return_value = 100
            param.requires_grad = True
            mock_params.append(param)
        
        # 2 non-trainable parameters (should not be counted)
        for i in range(2):
            param = MagicMock()
            param.numel.return_value = 50
            param.requires_grad = False
            mock_params.append(param)
        
        mock_model.parameters.return_value = mock_params
        
        # Mock the init_model method to return our custom model
        reconstructor.init_model = lambda: mock_model
        
        # Should only count trainable parameters: 3 * 100 = 300
        assert reconstructor.num_params == 300
    
    def test_num_params_handles_init_model_error(self):
        """Test that errors in init_model are handled gracefully."""
        dataset = self.create_sparse_static_dataset()
        reconstructor = MockNeuralReconstructor(dataset, dataset.domain)
        
        # Mock init_model to raise an exception
        def failing_init_model():
            raise RuntimeError("Model initialization failed")
        
        reconstructor.init_model = failing_init_model
        
        # Should return 0 when init_model fails
        assert reconstructor.num_params == 0
    
    def test_num_params_with_zero_parameters(self):
        """Test behavior with a model that has no parameters."""
        dataset = self.create_sparse_static_dataset()
        reconstructor = MockNeuralReconstructor(dataset, dataset.domain)
        
        # Mock model with no parameters
        mock_model = MagicMock()
        mock_model.parameters.return_value = []
        reconstructor.init_model = lambda: mock_model
        
        assert reconstructor.num_params == 0


def test_num_params_property_available_on_base_class():
    """Test that num_params property is available on BaseReconstructor."""
    assert hasattr(BaseReconstructor, 'num_params')
    
    # Check it's a property
    assert isinstance(getattr(BaseReconstructor, 'num_params'), property)