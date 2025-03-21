# 🌍 Project climatrix

This repository toolbox for sampling and reconstructing climate datasets. 
In particular, it contains [xarray](https://docs.xarray.dev/en/latest/index.html) accessor to 
facilitate usage.


---

## 👤 Author

- **Name:** Jakub Walczak
- **GitHub:** [@jamesWalczak](https://github.com/jamesWalczak)  
- **Email:** jakub.walczak@p.lodz.pl 

---

## 📌 Version

**Current Version:** `0.1a0` 🧪  

> [!CAUTION]  
> This is an alpha release – features are still evolving, and breaking changes may occur.




---

## 📚 Table of Contents

- [🚀 Getting Started](#-getting-started)
- [📦 Installation](#-installation)
- [⚙️ Usage](#️-usage)
- [🧪 Examples](#-examples)
- [🛠️ Features](#️-features)
- [📄 License](#-license)
- [🙏 Acknowledgements](#-acknowledgements)

---

## 🚀 Getting Started

These instructions will get you a copy of the project up and running on your local machine.

```bash
git clone https://github.com/yourusername/my-python-project.git
cd my-python-project
```

> [!IMPORTANT]  
> The project soon will be available via PyPI (`pip install ...`)

---

## ⚙️ Usage

Here is a basic example of how to use this project:

```python
# TODO
```

---

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

---


## 🛠️ Features

- 🧭 Easy access to coordinate data (similar to MetPy), using regex to locate lat/lon
- 📊 Sampling of climate data, both **uniformly** and using **normal-like distributions**
- 🔁 Reconstruction via:
  - **IDW** (Inverse Distance Weighting)
  - **Ordinary Kriging**
  - **SIREN** (Sinusoidal Representation Networks)
- 🧪 Tools to compare reconstruction results
- 📈 Plotting utilities for visualizing inputs and outputs


---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

 to be done.

---
