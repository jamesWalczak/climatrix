# 🌍 climatrix

<div align="center">

<img src="https://raw.githubusercontent.com/jamesWalczak/climatrix/0e2a3ab98836642140e50f2e59e314134c61137f/docs/assets/logo.svg" width="200" alt="Climatrix Logo">

<br>

<!-- Badges -->

<a href="https://www.python.org/downloads">
  <img src="https://img.shields.io/badge/-Python_3.12%7C3.13-blue?logo=python&logoColor=white" alt="Python Versions">
</a>
<a href="https://black.readthedocs.io/en/stable/">
  <img src="https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray" alt="Code Style: Black">
</a>
<a href="https://pycqa.github.io/isort/">
  <img src="https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336" alt="Import Sort: isort">
</a>
<a href="https://github.com/jamesWalczak/climatrix/blob/main/LICENSE">
  <img src="https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray" alt="License: MIT">
</a>

</div>

______________________________________________________________________

**Climatrix** is a flexible toolbox for sampling and reconstructing climate datasets.

It provides utilities and an [xarray](https://docs.xarray.dev/en/latest/index.html) accessor that simplifies the workflow of working with climate data arrays — from preprocessing to statistical sampling.

______________________________________________________________________

## 👤 Author

- **Name:** Jakub Walczak
- **GitHub:** [@jamesWalczak](https://github.com/jamesWalczak)
- **Email:** jakub.walczak@p.lodz.pl

______________________________________________________________________

## 👥 Contributors

- **Name:** Wojciech Żyndul
- **GitHub:** [@wzyndul](https://github.com/wzyndul)
- **Email:** 242575@edu.p.lodz.pl

______________________________________________________________________

## 📌 Version

> **Important**
> This is an alpha release – features are still evolving, and breaking changes may occur.

______________________________________________________________________

## 📚 Table of Contents

- [🚀 Getting Started](#-getting-started)
- [📦 Installation](#-installation)
- [⚙️ Usage](#%EF%B8%8F-usage)
- [🧪 Examples](#-examples)
- [🛠️ Features](#%EF%B8%8F-features)
- [📄 License](#-license)
- [🙏 Citation](#-citation)

______________________________________________________________________

## ⚙️ Usage

Getting started and API reference are available in the official [documentation](https://jameswalczak.github.io/climatrix/latest/).

______________________________________________________________________

## 🧪 Examples

<details>
<summary>🔍 Click to expand example: Accessing `climatrix` features</summary>

```python
import climatrix as cm
import xarray as xr

my_dataset = "/file/to/netcdf.nc
cm_dset = xr.open_dataset(my_dataset).cm
```

</details>

<details>
<summary>📊 Click to expand example: Getting values of coordinate</summary>

```python
import climatrix as cm
import xarray as xr

my_dataset = "/file/to/netcdf.nc"
cm_dset = xr.open_dataset(my_dataset).cm
print("Latitude values: ", cm_dset.latitude)
print("Time values: ", cm_dset.time)
```

</details>

<details>
<summary>📊 Subsetting by bounding box</summary>

```python
import climatrix as cm
import xarray as xr

my_dataset = "/file/to/netcdf.nc
cm_dset = xr.open_dataset(my_dataset).cm
europe = cm_dset.cm.subset(north=71, south=36, west=-24, east=35)
```

</details>

______________________________________________________________________

## 🛠️ Features

- 🧭 Easy access to coordinate data (similar to MetPy), using regex to locate lat/lon
- 📊 Sampling of climate data, both **uniformly** and using **normal-like distributions**
- 🔁 Reconstruction via:
  - **IDW** (Inverse Distance Weighting)
  - **Ordinary Kriging**
  - **SIREN** (Sinusoidal INR)
- 🧪 Tools to compare reconstruction results
- 📈 Plotting utilities for visualizing inputs and outputs
- 🔧 Hyperparameter Optimization

______________________________________________________________________

## 🔧 Hyperparameter Optimization

Climatrix provides automated hyperparameter optimization for all reconstruction methods using Bayesian optimization. The `HParamFinder` class offers an intuitive interface for finding optimal parameters.

### Quick Start

```python
from climatrix.optim import HParamFinder

# Basic usage - optimize IDW parameters
finder = HParamFinder(train_dataset, validation_dataset, method="idw")
result = finder.optimize()
best_params = result['best_params']

# Use optimized parameters for reconstruction
optimized_reconstruction = train_dataset.reconstruct(
    target=test_domain,
    method="idw", 
    **best_params
)
```

### Advanced Usage

```python
# Optimize specific parameters only
finder = HParamFinder(
    train_dataset, validation_dataset,
    method="sinet",
    include=["lr", "batch_size"],     # Only optimize these parameters
    exclude=["k"],                    # Or exclude specific parameters  
    metric="rmse",                    # Optimization metric (mae, mse, rmse)
    explore=0.7,                      # Exploration vs exploitation (0-1)
    n_iters=50,                       # Total optimization iterations
    random_seed=123                   # For reproducible results
)

result = finder.optimize()
print(f"Best parameters: {result['best_params']}")
print(f"Best {result['metric_name']} score: {result['best_score']}")
```

### Installation

The hyperparameter optimization feature requires the `bayesian-optimization` package:

```bash
pip install climatrix[optim]
```

______________________________________________________________________

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/jamesWalczak/climatrix/blob/main/LICENSE) file for details.

## 👥 Contributing

The rules for contributing on the project are described in [CONTRIBUTING](https://github.com/jamesWalczak/climatrix/blob/main/CONTRIBUTING.md) file in details.

______________________________________________________________________

## 🙏 Citation

If you are using this software in scientific work, cite us:

```
@article{walczak2025climatrix,
  title={Climatrix: Xarray accessor for climate data sampling and reconstruction},
  author={Walczak, Jakub and {\.Z}yndul, Wojciech},
  journal={SoftwareX},
  volume={31},
  pages={102263},
  year={2025},
  publisher={Elsevier}
}
```
