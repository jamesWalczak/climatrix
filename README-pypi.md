# 🌍 Project climatrix

This repository toolbox for sampling and reconstructing climate datasets.
In particular, it contains [xarray](https://docs.xarray.dev/en/latest/index.html) accessor to
facilitate usage.

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

**Current Version:** `0.1a0` 🧪

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

______________________________________________________________________

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Contributing

The rules for contributing on the project are described in [CONTRIBUTING](CONTRIBUTING.md) file in details.

______________________________________________________________________

## 🙏 Citation

If you are using this software in scientific work, cite us:

```bibtex
@misc{climatrix,
  author       = {Walczak, J., Żyndul, W.},
  title        = {climatrix: Climate data reconstruction made simple },
  year         = {2025},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/jamesWalczak/climatrix}},
}