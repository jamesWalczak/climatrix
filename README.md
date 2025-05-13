# 🌍 Project climatrix

<div align="center">
<img src="https://github.com/jamesWalczak/climatrix/blob/0e2a3ab98836642140e50f2e59e314134c61137f/docs/assets/logo.svg" width="20%" height="20%">

# A quick way to start with machine and deep learning

[![python](https://img.shields.io/badge/-Python_3.12%7C3.13-blue?logo=python&logoColor=white)](https://www.python.org/downloads)

[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/jamesWalczak/climatrix/blob/main/LICENSE)

[![PyPI version](https://badge.fury.io/py/climatrix.svg)](https://badge.fury.io/py/climatrix)

</div>

This repository toolbox for sampling and reconstructing climate datasets.
In particular, it contains [xarray](https://docs.xarray.dev/en/latest/index.html) accessor to
facilitate usage.

______________________________________________________________________

## 👤 Author

- **Name:** Jakub Walczak
- **GitHub:** [@jamesWalczak](https://github.com/jamesWalczak)
- **Email:** jakub.walczak@p.lodz.pl

______________________________________________________________________

## 📌 Version

**Current Version:** `0.1a0` 🧪

> [!CAUTION]\
> This is an alpha release – features are still evolving, and breaking changes may occur.

______________________________________________________________________

## 📚 Table of Contents

- [🚀 Getting Started](#-getting-started)
- [📦 Installation](#-installation)
- [⚙️ Usage](#%EF%B8%8F-usage)
- [🧪 Examples](#-examples)
- [🛠️ Features](#%EF%B8%8F-features)
- [📄 License](#-license)
- [🙏 Acknowledgements](#-acknowledgements)

______________________________________________________________________

## 🚀 Getting Started

These instructions will get you a copy of the project up and running on your local machine.

```bash
git clone https://github.com/jamesWalczak/climatrix/
cd climatrix
```

> [!IMPORTANT]\
> The project soon will be available via PyPI (`pip install ...`)

______________________________________________________________________

## ⚙️ Usage

Here is a basic example of how to use this project:

```python
# TODO
```

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
# TODO
```

</details>

______________________________________________________________________

## 🛠️ Features

- 🧭 Easy access to coordinate data (similar to MetPy), using regex to locate lat/lon
- 📊 Sampling of climate data, both **uniformly** and using **normal-like distributions**
- 🔁 Reconstruction via:
  - **IDW** (Inverse Distance Weighting)
  - **Ordinary Kriging**
  - **SiNET** (Sinusoidal reconstruction)
- 🧪 Tools to compare reconstruction results
- 📈 Plotting utilities for visualizing inputs and outputs

______________________________________________________________________

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/jamesWalczak/climatrix/blob/main/LICENSE) file for details.

## 👥 Contributing

The rules for contributing on the project are described in [CONTRIBUTING](CONTRIBUTING.md) file in details.

______________________________________________________________________

## 🙏 Acknowledgements

to be done.

______________________________________________________________________
