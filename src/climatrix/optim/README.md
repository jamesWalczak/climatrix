# Hyperparameter Optimization

This module provides Bayesian hyperparameter optimization for reconstruction methods in Climatrix.

## Overview

The `HParamFinder` class uses Bayesian optimization to automatically find the best hyperparameters for various reconstruction methods, including:

- **IDW (Inverse Distance Weighting)**: `power`, `k`, `k_min`
- **OK (Ordinary Kriging)**: `nlags`, `weight`, `verbose`, `pseudo_inv`
- **SiNET**: `lr`, `batch_size`, `num_epochs`, `gradient_clipping_value`, `mse_loss_weight`, `eikonal_loss_weight`, `laplace_loss_weight`
- **SIREN**: `lr`, `batch_size`, `num_epochs`, `hidden_dim`, `num_layers`, `gradient_clipping_value`

## Installation

To use the optimization features, install the optional dependency:

```bash
pip install climatrix[optim]
```

Or install the bayesian-optimization package directly:

```bash
pip install bayesian-optimization
```

## Basic Usage

```python
from climatrix.optim import HParamFinder

# Create HParamFinder
finder = HParamFinder(
    train_dset,     # Training dataset
    val_dset,       # Validation dataset
    metric="mae",   # Optimization metric (mae, mse, rmse)
    method="idw",   # Reconstruction method
    explore=0.9,    # Exploration vs exploitation (0 < explore < 1)
    n_iters=100     # Total optimization iterations
)

# Run optimization
result = finder.optimize()

# Get best parameters
best_params = result['best_params']
best_score = result['best_score']
```

## Advanced Usage

### Parameter Selection

```python
# Only optimize specific parameters
finder = HParamFinder(
    train_dset, val_dset,
    include=["power", "k"]  # Only optimize these
)

# Exclude specific parameters
finder = HParamFinder(
    train_dset, val_dset,
    exclude="k_min"  # Don't optimize this
)
```

### Custom Bounds

```python
# Override default parameter bounds
custom_bounds = {
    "power": (1.0, 3.0),
    "k": (2, 8)
}

finder = HParamFinder(
    train_dset, val_dset,
    bounds=custom_bounds
)
```

### Different Methods

```python
# Optimize SiNET parameters
finder = HParamFinder(
    train_dset, val_dset,
    method="sinet",
    metric="rmse"
)

# Optimize Kriging parameters
finder = HParamFinder(
    train_dset, val_dset,
    method="ok",
    include=["nlags", "weight"]
)
```

## Optimization Parameters

- **explore** (float): Controls exploration vs exploitation trade-off
  - Higher values (closer to 1.0) favor exploration of new parameter regions
  - Lower values favor exploitation of promising regions
  - Range: (0, 1)

- **n_iters** (int): Total number of optimization iterations
  - Split between initial random sampling and Bayesian optimization
  - `n_init_points = int(n_iters * explore)`
  - `n_iter = n_iters - n_init_points`

## Output Format

The `optimize()` method returns a dictionary containing:

```python
{
    'best_params': {...},      # Best hyperparameters found
    'best_score': float,       # Best score (negative metric value)
    'metric_name': str,        # Metric used for optimization
    'method': str,             # Reconstruction method
    'history': [...]           # Optimization history
}
```

## Example

See `examples/hyperparameter_optimization.py` for a complete example demonstrating various usage patterns.

## API Reference

### HParamFinder

Class for Bayesian hyperparameter optimization.

#### Constructor

```python
HParamFinder(
    train_dset: BaseClimatrixDataset,
    val_dset: BaseClimatrixDataset,
    *,
    metric: str = "mae",
    method: str = "idw", 
    exclude: Union[str, Collection[str], None] = None,
    include: Union[str, Collection[str], None] = None,
    explore: float = 0.9,
    n_iters: int = 100,
    bounds: dict[str, tuple[float, float]] | None = None,
)
```

#### Methods

- `optimize() -> dict`: Run Bayesian optimization and return results

### Functions

- `get_hparams_bounds(method: str) -> dict`: Get default hyperparameter bounds for a method