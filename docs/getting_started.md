# 🚀 Getting Started with `climatrix`

Welcome to **climatrix** – a Python library designed for efficient sampling and reconstruction of climate datasets. This guide will help you set up and start using `climatrix` effectively.

---

## 📦 Installation

### 🔧 Prerequisites

Ensure you have the following installed:

- **Python 3.12 or higher**
- **pip** (Python package installer)

### 🛠️ Installing `climatrix`

You can install `climatrix` directly from GitHub:

```bash
pip install git+https://github.com/jamesWalczak/climatrix.git
```

???+ info "Coming soon on PyPI"

    The project will soon be available via PyPI (`pip install ...`)



## 🧪 Verifying the Installation

To confirm that `climatrix` is installed correctly, run the following in your Python environment:


```python
import climatrix as cm

print(cm.__version__)
```

## 🔍 Exploring `climatrix`

The core functionality of `climatrix` revolves around the [`BaseClimatrixDataset`](api.md#climatrix.dataset.base.BaseClimatrixDataset) and [`Domain`](api.md#climatrix.dataset.domain.Domain) classes, which provides methods for:

- [Accessing spatio-temporal axes](#accessing-spatio-temporal-axes)
- [Subsetting datasets](#subsetting-dataset-by-geographical-coordinates) based on geographic bounds,
- [Selecting time](#selecting-time),
- [Sampling data](#sampling-data) using uniform or normal distributions,
- [Reconstructing](#reconstructing) datasets from samples,
- [Plotting](#plotting) data for visualization.

### Creating [`BaseClimatrixDataset`](api.md#climatrix.dataset.base.BaseClimatrixDataset)

You can create [`BaseClimatrixDataset`](api.md#climatrix.dataset.base.BaseClimatrixDataset) directly, by passing `xarray.DataArray` or `xarray.Daaset` to the initializer:

???+ note
    In the current version, `climatrix` supports only static (single-element or no time dimension) and single-variable datasets.

    It means, [`BaseClimatrixDataset`](api.md#climatrix.dataset.base.BaseClimatrixDataset) can be created based
    on `xarray.DataArray` or single-variable `xarray.Dataset`.

```python
import climatrix as cm

dset = cm.BaseClimatrixDataset(xarray_dataset)
```

but `climatrix` was implemented as `xarray` accessor, so there is more convenient way to create [`BaseClimatrixDataset`](api.md#climatrix.dataset.base.BaseClimatrixDataset):

```python
import climatrix as cm

dset = xarray_dataset.cm
```

???+ warning
    When using `climatrix` as accessor, remember to import `climatrix` first!

### Accessing spatio-temporal axes

### Subsetting dataset by geographical coordinates

### Selecting time

### Sampling data

### Reconstructing

### Plotting

