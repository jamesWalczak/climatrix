"""
Example usage of HParamFinder for Bayesian hyperparameter optimization.

This example demonstrates how to use the HParamFinder class to optimize
hyperparameters for different reconstruction methods.
"""

import numpy as np
import xarray as xr
from datetime import datetime

from climatrix import BaseClimatrixDataset
from climatrix.optim import HParamFinder

def create_example_datasets():
    """Create example sparse and dense datasets for demonstration."""
    
    # Create a sparse dataset (training data)
    np.random.seed(42)
    sparse_data = np.random.rand(10, 1) * 10 + 20  # Temperature-like data
    sparse_coords = {
        "point": np.arange(10),
        "time": np.array([datetime(2000, 1, 1)], dtype="datetime64"),
        "latitude": (("point",), np.linspace(-80, 80, 10)),
        "longitude": (("point",), np.linspace(-150, 150, 10)),
    }
    
    sparse_dataset = BaseClimatrixDataset(
        xr.DataArray(
            data=sparse_data,
            dims=("point", "time"),
            coords=sparse_coords,
            name="temperature"
        )
    )
    
    # Create a dense dataset (validation data)
    dense_data = np.random.rand(1, 5, 5) * 8 + 22  # Similar temperature data
    dense_coords = {
        "time": np.array([datetime(2000, 1, 1)], dtype="datetime64"),
        "latitude": (("latitude",), np.linspace(-60, 60, 5)),
        "longitude": (("longitude",), np.linspace(-120, 120, 5)),
    }
    
    dense_dataset = BaseClimatrixDataset(
        xr.DataArray(
            data=dense_data,
            dims=("time", "latitude", "longitude"),
            coords=dense_coords,
            name="temperature"
        )
    )
    
    return sparse_dataset, dense_dataset

def example_basic_usage():
    """Demonstrate basic usage of HParamFinder."""
    print("=== Basic Usage Example ===")
    
    train_dset, val_dset = create_example_datasets()
    
    # Create HParamFinder with default settings
    finder = HParamFinder(train_dset, val_dset)
    
    print(f"Method: {finder.method}")
    print(f"Metric: {finder.metric}")
    print(f"Random seed: {finder.random_seed}")
    print(f"Parameters to optimize: {list(finder.bounds.keys())}")
    print(f"Number of initialization points: {finder.n_init_points}")
    print(f"Number of optimization iterations: {finder.n_iter}")
    
    # Note: optimize() would require bayesian-optimization package
    # result = finder.optimize()
    # print(f"Best parameters: {result['best_params']}")
    print()

def example_custom_parameters():
    """Demonstrate usage with custom parameters."""
    print("=== Custom Parameters Example ===")
    
    train_dset, val_dset = create_example_datasets()
    
    # Optimize only specific parameters with custom settings
    finder = HParamFinder(
        train_dset, 
        val_dset,
        metric="mse",
        method="idw",
        include=["power", "k"],  # Only optimize these parameters
        explore=0.7,  # More exploitation vs exploration
        n_iters=50,   # Fewer iterations
        bounds={"power": (1.0, 3.0), "k": (3, 10)},  # Custom bounds
        random_seed=123  # Custom random seed
    )
    
    print(f"Method: {finder.method}")
    print(f"Metric: {finder.metric}")
    print(f"Random seed: {finder.random_seed}")
    print(f"Parameters to optimize: {list(finder.bounds.keys())}")
    print(f"Custom bounds: {finder.bounds}")
    print(f"Exploration ratio: {finder.n_init_points / (finder.n_init_points + finder.n_iter)}")
    print()

def example_exclude_parameters():
    """Demonstrate usage with parameter exclusion."""
    print("=== Parameter Exclusion Example ===")
    
    train_dset, val_dset = create_example_datasets()
    
    # Exclude certain parameters from optimization
    finder = HParamFinder(
        train_dset,
        val_dset,
        method="idw",
        exclude=["k_min"]  # Don't optimize this parameter
    )
    
    print(f"Method: {finder.method}")
    print(f"All IDW parameters: {list(get_hparams_bounds('idw').keys())}")
    print(f"Parameters to optimize (excluding k_min): {list(finder.bounds.keys())}")
    print()

def example_include_exclude_both():
    """Demonstrate usage with both include and exclude parameters."""
    print("=== Include and Exclude Both Example ===")
    
    train_dset, val_dset = create_example_datasets()
    
    # Use both include and exclude (but no common parameters)
    finder = HParamFinder(
        train_dset,
        val_dset,
        method="idw", 
        include=["power", "k", "k_min"],  # Start with these
        exclude=["k_min"]  # But exclude this one
    )
    
    print(f"Method: {finder.method}")
    print(f"All IDW parameters: {list(get_hparams_bounds('idw').keys())}")
    print(f"Final parameters to optimize: {list(finder.bounds.keys())}")
    print()

def example_different_methods():
    """Demonstrate usage with different reconstruction methods."""
    print("=== Different Methods Example ===")
    
    train_dset, val_dset = create_example_datasets()
    
    methods = ["idw", "ok", "sinet", "siren"]
    
    for method in methods:
        try:
            bounds = get_hparams_bounds(method)
            print(f"{method.upper()} parameters: {list(bounds.keys())}")
            
            # Create finder for this method
            finder = HParamFinder(
                train_dset,
                val_dset,
                method=method,
                n_iters=20  # Small number for demo
            )
            print(f"  - Optimization setup ready for {method}")
            
        except Exception as e:
            print(f"  - Error with {method}: {e}")
    print()

def example_parameter_evaluation():
    """Demonstrate parameter evaluation without full optimization."""
    print("=== Parameter Evaluation Example ===")
    
    train_dset, val_dset = create_example_datasets()
    
    finder = HParamFinder(train_dset, val_dset, method="idw")
    
    # Test different parameter combinations
    test_params = [
        {"power": 1.0, "k": 3, "k_min": 1},
        {"power": 2.0, "k": 5, "k_min": 2},
        {"power": 3.0, "k": 8, "k_min": 3},
    ]
    
    print("Testing parameter combinations:")
    for i, params in enumerate(test_params):
        try:
            score = finder._evaluate_params(**params)
            print(f"  Params {i+1}: {params} -> Score: {score:.4f}")
        except Exception as e:
            print(f"  Params {i+1}: {params} -> Error: {e}")
    print()

if __name__ == "__main__":
    print("HParamFinder Usage Examples")
    print("=" * 50)
    
    # Import the get_hparams_bounds function to make it available
    from climatrix.optim.bayesian import get_hparams_bounds
    
    example_basic_usage()
    example_custom_parameters()
    example_exclude_parameters()
    example_include_exclude_both()
    example_different_methods()
    example_parameter_evaluation()
    
    print("Note: To run actual optimization, install the 'bayesian-optimization' package:")
    print("  pip install bayesian-optimization")
    print("Then call finder.optimize() to get optimized hyperparameters.")