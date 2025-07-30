# Hyperparameter Optimization

Climatrix provides automated hyperparameter optimization for all reconstruction methods using Bayesian optimization. This feature helps you find optimal parameters without manual tuning.

## Overview

The `HParamFinder` class uses Bayesian optimization to efficiently search the hyperparameter space and find parameter values that minimize reconstruction error on a validation dataset.

## Basic Usage

```python
from climatrix.optim import HParamFinder

# Load your training and validation datasets
train_dset = ...  # Your training dataset
val_dset = ...    # Your validation dataset

# Create the hyperparameter finder
finder = HParamFinder(train_dset, val_dset, method="idw")

# Run optimization
result = finder.optimize()

# Access the best parameters
best_params = result['best_params']
best_score = result['best_score']

print(f"Best parameters: {best_params}")
print(f"Best MAE score: {best_score}")
```

## Advanced Configuration

### Parameter Selection

You can control which parameters to optimize using `include` and `exclude`:

```python
# Optimize only specific parameters
finder = HParamFinder(
    train_dset, val_dset,
    method="sinet",
    include=["lr", "batch_size"]  # Only optimize these
)

# Exclude specific parameters
finder = HParamFinder(
    train_dset, val_dset, 
    method="idw",
    exclude=["k_min"]  # Don't optimize this parameter
)

# You can use both include and exclude (but they cannot have common parameters)
finder = HParamFinder(
    train_dset, val_dset,
    method="sinet", 
    include=["lr", "batch_size", "num_epochs"],
    exclude=["gradient_clipping_value"]  # No overlap with include
)
```

### Optimization Metrics

Choose from different evaluation metrics:

```python
finder = HParamFinder(
    train_dset, val_dset,
    metric="rmse"  # Options: "mae", "mse", "rmse"
)
```

### Exploration vs Exploitation

Control the balance between exploration and exploitation:

```python
finder = HParamFinder(
    train_dset, val_dset,
    explore=0.8,    # Higher values favor exploration (0-1)
    n_iters=100     # Total optimization iterations
)
```

### Custom Parameter Bounds

Override default parameter bounds:

```python
custom_bounds = {
    "power": (1.0, 3.0),  # Custom range for power parameter
    "k": (5, 15)          # Custom range for k parameter
}

finder = HParamFinder(
    train_dset, val_dset,
    method="idw",
    bounds=custom_bounds
)
```

### Reproducible Results

Set a random seed for reproducible optimization:

```python
finder = HParamFinder(
    train_dset, val_dset,
    random_seed=42
)
```

## Supported Methods and Parameters

### IDW (Inverse Distance Weighting)
- `power`: (0.5, 5.0) - Power parameter for distance weighting
- `k`: (1, 20) - Number of nearest neighbors
- `k_min`: (1, 10) - Minimum number of neighbors required

### Ordinary Kriging
- `nlags`: (4, 20) - Number of lags for variogram
- `weight`: (0.0, 1.0) - Weight parameter
- `verbose`: (0, 1) - Verbosity level (boolean)
- `pseudo_inv`: (0, 1) - Use pseudo-inverse (boolean)

### SiNET (Spatial Interpolation NET)
- `lr`: (1e-5, 1e-2) - Learning rate
- `batch_size`: (64, 1024) - Batch size for training
- `num_epochs`: (1000, 10000) - Number of training epochs
- `gradient_clipping_value`: (0.1, 10.0) - Gradient clipping threshold
- `mse_loss_weight`: (1e1, 1e4) - Weight for MSE loss
- `eikonal_loss_weight`: (1e0, 1e3) - Weight for Eikonal loss
- `laplace_loss_weight`: (1e1, 1e3) - Weight for Laplace loss

### SIREN (Sinusoidal INR)
- `lr`: (1e-5, 1e-2) - Learning rate
- `batch_size`: (64, 1024) - Batch size for training
- `num_epochs`: (1000, 10000) - Number of training epochs
- `hidden_dim`: (128, 512) - Hidden layer dimensions
- `num_layers`: (3, 8) - Number of hidden layers
- `gradient_clipping_value`: (0.1, 10.0) - Gradient clipping threshold

## Complete Example

```python
from climatrix.optim import HParamFinder
import climatrix as cm

# Load datasets
train_dset = cm.load_dataset("training_data.nc")
val_dset = cm.load_dataset("validation_data.nc")
test_dset = cm.load_dataset("test_data.nc")

# Optimize IDW parameters
finder = HParamFinder(
    train_dset, val_dset,
    method="idw",
    metric="rmse",
    exclude=["k_min"],  # Use default k_min
    explore=0.7,
    n_iters=50,
    random_seed=42
)

# Run optimization
result = finder.optimize()

print("Optimization Results:")
print(f"Best parameters: {result['best_params']}")
print(f"Best RMSE: {result['best_score']}")
print(f"Method: {result['method']}")

# Apply optimized parameters to test data
optimized_reconstruction = train_dset.reconstruct(
    target=test_dset.domain,
    method="idw",
    **result['best_params']
)

# Evaluate final performance
from climatrix.comparison import Comparison
comparison = Comparison(optimized_reconstruction, test_dset)
final_rmse = comparison.compute("rmse")
print(f"Final test RMSE: {final_rmse}")
```

## Installation

To use hyperparameter optimization, install climatrix with the optimization extras:

```bash
pip install climatrix[optim]
```

This installs the required `bayesian-optimization` package dependency.

## Performance Tips

1. **Start with fewer iterations** for quick testing, then increase for final optimization
2. **Use parameter filtering** (`include`/`exclude`) to focus on the most important parameters
3. **Higher exploration values** (0.8-0.9) work well for initial searches
4. **Lower exploration values** (0.3-0.5) can help fine-tune around good regions
5. **Set a random seed** for reproducible results during development