# Hyperparameter Optimization for Reconstruction Methods

This module provides Bayesian optimization for hyperparameter tuning of reconstruction methods in the Climatrix package.

## Installation

To use hyperparameter optimization, install the optional optimization dependencies:

```bash
pip install climatrix[optim]
# or
pip install bayesian-optimization
```

## Quick Start

```python
from climatrix.optim.bayesian import HParamFinder

# Create hyperparameter finder
finder = HParamFinder(
    train_dset=train_dataset,
    val_dset=validation_dataset,
    metric="mae",
    method="idw",
    explore=0.9,
    n_iters=100
)

# Run optimization
result = finder.optimize()
print(f"Best parameters: {result['best_params']}")
print(f"Best score: {result['best_score']}")
```

## Interface Specification

The `HParamFinder` class follows this exact interface as specified in the requirements:

```python
finder = HParamFinder(
    train_dset,              # BaseClimatrixDataset for training
    val_dset,                # BaseClimatrixDataset for validation  
    metric="mae",            # Evaluation metric
    method="idw",            # Reconstruction method
    exclude="k",             # Parameters to exclude (str or list)
    include=["power", "k_min"],  # Parameters to include (str or list)
    explore=0.9,             # Exploration/exploitation tradeoff (0 < explore < 1)
    n_iters=100              # Total optimization iterations
)
```

## Parameters

- **train_dset**: Training dataset (BaseClimatrixDataset)
- **val_dset**: Validation dataset (BaseClimatrixDataset)
- **metric**: Evaluation metric ("mae", "mse", "rmse", "r2")
- **method**: Reconstruction method ("idw", "kriging")
- **exclude**: Hyperparameters to exclude from optimization
- **include**: Hyperparameters to include (if specified, only these are optimized)
- **explore**: Exploration vs exploitation tradeoff (0 < explore < 1)
- **n_iters**: Total number of optimization iterations
- **kwargs**: Custom bounds for hyperparameters (e.g., `power=(1.0, 3.0)`)

## Supported Methods

### IDW (Inverse Distance Weighting)

Default hyperparameter bounds:
- `power`: (0.5, 5.0) - Inverse distance weighting power
- `k`: (3, 20) - Number of nearest neighbors
- `k_min`: (1, 10) - Minimum number of neighbors

### Kriging

Default hyperparameter bounds are method-specific and depend on the variogram model used.

## Examples

### Basic Optimization

```python
from climatrix.optim.bayesian import HParamFinder

finder = HParamFinder(train_data, val_data, method="idw")
result = finder.optimize()
```

### Selective Parameter Optimization

```python
# Only optimize power and k, exclude k_min
finder = HParamFinder(
    train_data, val_data,
    method="idw",
    include=["power", "k"],
    n_iters=50
)
result = finder.optimize()
```

### Custom Parameter Bounds

```python
# Override default bounds
finder = HParamFinder(
    train_data, val_data,
    method="idw", 
    power=(1.0, 3.0),  # Custom bounds for power
    k=(5, 15)          # Custom bounds for k
)
result = finder.optimize()
```

### Inspecting Default Bounds

```python
from climatrix.optim.bayesian import get_hparams_bounds

# Get default bounds for a method
bounds = get_hparams_bounds("idw")
print(bounds)  # {'power': (0.5, 5.0), 'k': (3, 20), 'k_min': (1, 10)}
```

## Advanced Usage

### Different Metrics

The following evaluation metrics are supported:

- **mae**: Mean Absolute Error (minimize)
- **mse**: Mean Squared Error (minimize)  
- **rmse**: Root Mean Squared Error (minimize)
- **r2**: R-squared (maximize)

### Exploration vs Exploitation

The `explore` parameter controls the balance between exploration and exploitation:

- Higher values (closer to 1.0): More exploration, fewer optimization iterations
- Lower values (closer to 0.0): Less exploration, more optimization iterations

The number of random initialization points is calculated as:
```python
init_points = max(1, int(n_iters * explore))
n_iter = n_iters - init_points
```

### Return Values

The `optimize()` method returns a dictionary with:

- **best_params**: Dictionary of best hyperparameters found
- **best_score**: Best validation score achieved
- **optimizer**: BayesianOptimization object for further analysis

## Implementation Notes

- Parameter type conversion is handled automatically (e.g., k and k_min are converted to integers)
- Failed evaluations return very poor scores rather than raising exceptions
- NaN values in validation data are handled gracefully
- The implementation uses proper error handling for missing dependencies

## Testing

The module includes comprehensive unit tests and integration tests that can run without the bayesian-optimization package installed. Run tests with:

```bash
python -m pytest tests/unit/optim/
```