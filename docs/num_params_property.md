# num_params Property for NN Reconstructors

This document demonstrates the new `num_params` property that has been added to all reconstructors in the climatrix package.

## Overview

The `num_params` property provides a convenient way to get the number of trainable parameters in neural network-based reconstructors. This is useful for:

- Comparing model complexity across different NN reconstructors
- Monitoring parameter count when tuning hyperparameters
- Research and optimization studies
- Understanding memory and computational requirements

## Usage

The property is available on all reconstructor instances:

```python
from climatrix import BaseClimatrixDataset
from climatrix.reconstruct import SiNETReconstructor, IDWReconstructor

# For neural network reconstructors
sinet = SiNETReconstructor(dataset, target_domain, hidden_dim=128)
print(f"SiNET parameters: {sinet.num_params}")  # e.g., 8576

# For traditional reconstructors  
idw = IDWReconstructor(dataset, target_domain)
print(f"IDW parameters: {idw.num_params}")  # 0
```

## Implementation Details

- **Location**: Property is defined in `BaseReconstructor` class
- **Detection**: Automatically detects NN vs traditional reconstructors by checking for `init_model` method
- **Parameter Counting**: Only counts trainable parameters (`requires_grad=True`)
- **Error Handling**: Returns 0 if model initialization fails
- **Performance**: Creates temporary model instance only when property is accessed

## Supported Reconstructors

### Neural Network Reconstructors (return parameter count):
- SiNETReconstructor
- SIRENReconstructor  
- MMGNReconstructor
- Any future NN-based reconstructors with `init_model` method

### Traditional Reconstructors (return 0):
- IDWReconstructor
- KrigingReconstructor (OK)
- Any reconstructors without neural networks

## Example Output

```python
# Different NN reconstructors with same input size
sinet_small = SiNETReconstructor(dataset, target_domain, hidden_dim=32)
sinet_large = SiNETReconstructor(dataset, target_domain, hidden_dim=256)

print(f"Small SiNET: {sinet_small.num_params} parameters")
print(f"Large SiNET: {sinet_large.num_params} parameters")

# Compare different architectures
siren = SIRENReconstructor(dataset, target_domain, hidden_features=128)
mmgn = MMGNReconstructor(dataset, target_domain, hidden_dim=128)

print(f"SIREN: {siren.num_params} parameters")  
print(f"MMGN: {mmgn.num_params} parameters")
```

## Integration with Hyperparameter Optimization

The property is particularly useful when optimizing architectures:

```python
def evaluate_architecture(hidden_dim):
    reconstructor = SiNETReconstructor(dataset, target_domain, hidden_dim=hidden_dim)
    
    param_count = reconstructor.num_params
    if param_count > max_parameters:
        return float('inf')  # Skip if too large
    
    # Continue with reconstruction and evaluation
    result = reconstructor.reconstruct()
    return compute_error(result, ground_truth)
```

## Notes

- The property creates a temporary model to count parameters, so there's minimal overhead
- Parameter counting excludes frozen/non-trainable parameters
- Gracefully handles model initialization errors
- Compatible with all existing and future reconstructor implementations