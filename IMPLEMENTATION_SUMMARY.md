# Implementation Summary: Bayesian Hyperparameter Optimization

## Overview

This implementation successfully addresses issue #85 by creating a complete Bayesian hyperparameter optimization module for reconstruction methods in Climatrix.

## Files Created

### Core Implementation
- `src/climatrix/optim/__init__.py` - Module initialization
- `src/climatrix/optim/bayesian.py` - Main implementation with HParamFinder class
- `src/climatrix/optim/README.md` - Comprehensive documentation

### Testing
- `tests/unit/optim/__init__.py` - Test module initialization  
- `tests/unit/optim/test_bayesian.py` - Comprehensive unit tests

### Examples
- `examples/hyperparameter_optimization.py` - Complete usage examples

## Key Features Implemented

### 1. HParamFinder Class
Exact interface as specified in the issue:
```python
finder = HParamFinder(
    train_dset,          # Training dataset
    val_dset,           # Validation dataset  
    metric="mae",       # Optimization metric
    method="idw",       # Reconstruction method
    exclude="k",        # Parameters to exclude
    include=["power", "k_min"],  # Parameters to include
    explore=0.9,        # Exploration vs exploitation
    n_iters=100         # Total iterations
)
```

### 2. Default Hyperparameter Bounds
Predefined bounds for all reconstruction methods:

- **IDW**: `power` (0.5-5.0), `k` (1-20), `k_min` (1-10)
- **Ordinary Kriging**: `nlags` (4-20), `weight` (0.0-1.0), `verbose` (0-1), `pseudo_inv` (0-1)
- **SiNET**: `lr` (1e-5 to 1e-2), `batch_size` (64-1024), `num_epochs` (1000-10000), plus loss weights
- **SIREN**: Similar to SiNET with additional `hidden_dim` (128-512), `num_layers` (3-8)

### 3. Flexible Parameter Selection
- **include**: Optimize only specified parameters
- **exclude**: Exclude specific parameters from optimization
- **bounds**: Override default bounds with custom values

### 4. Bayesian Optimization Integration
- Uses `bayesian-optimization` package (already in pyproject.toml)
- Computes `init_points` and `n_iter` based on `explore` parameter
- Handles parameter type conversion (int, bool, float)
- Returns comprehensive results with optimization history

### 5. Evaluation Metrics
- **MAE** (Mean Absolute Error) - default
- **MSE** (Mean Squared Error)  
- **RMSE** (Root Mean Squared Error)

### 6. Robust Error Handling
- Input validation for all parameters
- Type checking for datasets
- Graceful handling of optimization failures
- Informative error messages

## Technical Implementation Details

### Parameter Processing
- Automatic conversion of float parameters to integers where needed (k, batch_size, etc.)
- Boolean parameter handling for kriging options
- Bounds validation and filtering

### Optimization Strategy
- Initial random sampling: `n_init_points = int(n_iters * explore)`
- Bayesian optimization: `n_iter = n_iters - n_init_points`
- Maximization of negative metric (since BayesianOptimization maximizes)

### Integration
- Works with existing `reconstruct()` method of BaseClimatrixDataset
- Supports all available reconstruction methods
- Maintains compatibility with existing codebase

## Usage Examples

### Basic Usage
```python
from climatrix.optim import HParamFinder

finder = HParamFinder(train_dset, val_dset)
result = finder.optimize()
best_params = result['best_params']
```

### Advanced Usage  
```python
finder = HParamFinder(
    train_dset, val_dset,
    method="sinet",
    metric="rmse", 
    include=["lr", "batch_size"],
    explore=0.7,
    n_iters=50,
    bounds={"lr": (1e-4, 1e-3)}
)
result = finder.optimize()
```

## Testing

Comprehensive unit tests cover:
- Parameter bounds validation
- Constructor argument handling
- Include/exclude parameter logic
- Type conversion
- Error conditions
- Mock optimization scenarios

## Documentation

- Complete API documentation in README
- Usage examples with different scenarios
- Integration instructions
- Parameter descriptions

## Dependencies

- Requires `bayesian-optimization` package (optional dependency)
- Uses existing climatrix infrastructure
- Compatible with all supported reconstruction methods

## Compliance with Issue Requirements

✅ Module location: `src/climatrix/optim/bayesian.py`  
✅ Uses `bayesian-optimization` package  
✅ Exact interface as specified  
✅ All reconstruction methods supported  
✅ Default bounds via `get_hparams_bounds()`  
✅ Include/exclude parameter support  
✅ Exploration/exploitation tradeoff  
✅ Proper init_points/n_iter computation  
✅ Unit test scenarios included  

The implementation is complete, well-tested, and ready for production use.