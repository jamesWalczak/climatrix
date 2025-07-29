"""
Example: Hyperparameter Optimization for Reconstruction Methods

This example demonstrates how to use the HParamFinder class to optimize
hyperparameters for reconstruction methods using Bayesian optimization.
"""

from datetime import datetime
import numpy as np
import xarray as xr

from climatrix import BaseClimatrixDataset
from climatrix.optim.bayesian import HParamFinder


def create_sample_data():
    """Create sample training and validation datasets."""
    
    # Create training dataset with sparse points
    train_data = BaseClimatrixDataset(
        xr.DataArray(
            data=np.random.rand(10, 1),
            dims=("point", "time"),
            coords={
                "point": np.arange(10),
                "time": np.array([datetime(2000, 1, 1)], dtype="datetime64"),
                "latitude": (("point",), np.linspace(-80, 80, 10)),
                "longitude": (("point",), np.linspace(-160, 160, 10)),
            },
        )
    )
    
    # Create validation dataset with different sparse points
    val_data = BaseClimatrixDataset(
        xr.DataArray(
            data=np.random.rand(5, 1),
            dims=("point", "time"),
            coords={
                "point": np.arange(5),
                "time": np.array([datetime(2000, 1, 1)], dtype="datetime64"),
                "latitude": (("point",), np.linspace(-60, 60, 5)),
                "longitude": (("point",), np.linspace(-120, 120, 5)),
            },
        )
    )
    
    return train_data, val_data


def example_basic_optimization():
    """Basic hyperparameter optimization example."""
    
    print("=== Basic Hyperparameter Optimization ===")
    
    # Create sample data
    train_dset, val_dset = create_sample_data()
    
    # Create hyperparameter finder with default settings
    finder = HParamFinder(
        train_dset=train_dset,
        val_dset=val_dset,
        metric="mae",
        method="idw",
        explore=0.9,
        n_iters=20  # Small number for example
    )
    
    print(f"Optimizing parameters: {list(finder.bounds.keys())}")
    print(f"Parameter bounds: {finder.bounds}")
    print(f"Init points: {finder.init_points}, Optimization iterations: {finder.n_iter}")
    
    # Run optimization (requires bayesian-optimization package)
    try:
        result = finder.optimize()
        
        print(f"\\nOptimization completed!")
        print(f"Best parameters: {result['best_params']}")
        print(f"Best score: {result['best_score']:.6f}")
        
        return result
        
    except ImportError as e:
        print(f"\\nOptimization requires bayesian-optimization package: {e}")
        return None


def example_selective_optimization():
    """Example with selective parameter optimization."""
    
    print("\\n=== Selective Parameter Optimization ===")
    
    # Create sample data
    train_dset, val_dset = create_sample_data()
    
    # Only optimize specific parameters
    finder = HParamFinder(
        train_dset=train_dset,
        val_dset=val_dset,
        metric="rmse",
        method="idw",
        include=["power", "k"],  # Only optimize power and k
        explore=0.7,
        n_iters=15
    )
    
    print(f"Optimizing parameters: {list(finder.bounds.keys())}")
    print(f"Parameter bounds: {finder.bounds}")
    
    try:
        result = finder.optimize()
        print(f"\\nBest parameters: {result['best_params']}")
        print(f"Best score: {result['best_score']:.6f}")
        
    except ImportError as e:
        print(f"\\nOptimization requires bayesian-optimization package: {e}")


def example_custom_bounds():
    """Example with custom parameter bounds."""
    
    print("\\n=== Custom Parameter Bounds ===")
    
    # Create sample data
    train_dset, val_dset = create_sample_data()
    
    # Use custom bounds for some parameters
    finder = HParamFinder(
        train_dset=train_dset,
        val_dset=val_dset,
        metric="mae",
        method="idw",
        exclude=["k_min"],  # Exclude k_min from optimization
        explore=0.8,
        n_iters=10,
        # Custom bounds override defaults
        power=(1.0, 3.0),  # Narrower range for power
        k=(5, 15)          # Different range for k
    )
    
    print(f"Optimizing parameters: {list(finder.bounds.keys())}")
    print(f"Parameter bounds (with custom overrides): {finder.bounds}")
    
    try:
        result = finder.optimize()
        print(f"\\nBest parameters: {result['best_params']}")
        print(f"Best score: {result['best_score']:.6f}")
        
    except ImportError as e:
        print(f"\\nOptimization requires bayesian-optimization package: {e}")


def example_different_metrics():
    """Example showing different evaluation metrics."""
    
    print("\\n=== Different Evaluation Metrics ===")
    
    train_dset, val_dset = create_sample_data()
    
    metrics = ["mae", "mse", "rmse", "r2"]
    
    for metric in metrics:
        print(f"\\nUsing metric: {metric}")
        
        finder = HParamFinder(
            train_dset=train_dset,
            val_dset=val_dset,
            metric=metric,
            method="idw",
            include=["power"],  # Just optimize power for quick example
            n_iters=5
        )
        
        print(f"  - Optimizing: {list(finder.bounds.keys())}")
        
        # Note: In practice, you would run optimization here
        print(f"  - Ready for optimization with {metric} metric")


if __name__ == "__main__":
    print("Climatrix Hyperparameter Optimization Examples")
    print("=" * 50)
    
    # Run examples
    example_basic_optimization()
    example_selective_optimization()
    example_custom_bounds()
    example_different_metrics()
    
    print("\\n" + "=" * 50)
    print("To run optimization, install: pip install bayesian-optimization")
    print("Then import and use HParamFinder as shown in the examples above.")