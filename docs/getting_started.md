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

???+ info "Climatrix is already available on PyPI"

    The project can be downloaded with `pip install climatrix`.



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

``` { .python .annotate }
import climatrix as cm # (1)!

dset = xarray_dataset.cm
```

1.  Even though, we don't use `climatrix` explicitly, we need to import climatrix to make `xarray` accessor visible.

???+ warning
    When using `climatrix` as accessor, remember to import `climatrix` first!

### Accessing spatio-temporal axes

By using [Climatrix](https://jameswalczak.github.io/climatrix/latest), you can easily acces spatio temporal axis. 

???+ info
    You don't need to know the name of axis (`lat`, `latitude` or anything else), [Climatrix](https://jameswalczak.github.io/climatrix/latest) automatically finds proper axis by matching regular expressions.


All predefined axis are available via [`Axis`](/climatrix/api#climatrix.dataset.domain.Axis) enum class.

To access latitude name, just use:

``` { .python }
xarray_dataset.cm.latitude.name
```

and to access values, use:

``` { .python }
xarray_dataset.cm.latitude.values
```

Below, you can find available attributes:

| Attribute | Meaning | 
| --------- | ------- | 
| `latitude`| [`Axis`](/climatrix/api#climatrix.dataset.axis.Axis) corresponding to `AxisType.LATITUDE` for the dataset |
| `longitude`| [`Axis`](/climatrix/api#climatrix.dataset.axis.Axis) corresponding to `AxisType.LONGITUDE` for the dataset  |
| `time` | [`Axis`](/climatrix/api#climatrix.dataset.axis.Axis) corresponding to `AxisType.TIME` for the dataset  |
| `point` | [`Axis`](/climatrix/api#climatrix.dataset.axis.Axis) corresponding to `AxisType.POINT` for the dataset  |
| `vertical` | [`Axis`](/climatrix/api#climatrix.dataset.axis.Axis) corresponding to `AxisType.VERTICAL` for the dataset  |



### Subsetting dataset by geographical coordinates
[Climatrix](https://jameswalczak.github.io/climatrix/latest) facilitates subsetting region based on bounding box.
To select Europe, just use the following command:

``` { .python }
europe = xarray_dataset.cm.subset(north=71, south=36, west=-24, east=35)
```

???+ warning
    If you attempt to select region not aligned with the dataset longitude convention, [Climatrix](https://jameswalczak.github.io/climatrix/latest) will inform you about it and ask for explicit update of convention.

???+ tip
    With [Climatrix](https://jameswalczak.github.io/climatrix/latest) chaning convention is easy!
    
    To switch to signed longitude convention ($\lambda \in [-180, 180]$) use [`to_signed_longitude`](https://jameswalczak.github.io/climatrix/latest/api/#climatrix.dataset.base.BaseClimatrixDataset.to_signed_longitude) method.

    ``` { .python }
    europe = xarray_dataset.cm.to_signed_longitude()
    ```

    To switch to positive-only longitude convention ($\lambda \in [0, 360]$), use [`to_positive_longitude`](https://jameswalczak.github.io/climatrix/latest/api/#climatrix.dataset.base.BaseClimatrixDataset.to_positive_longitude) method.

    ``` { .python }
    europe = xarray_dataset.cm.to_positive_longitude()
    ```    

### Selecting time

You can select time instants by integer (indices on time axis):

``` { .python .copy .annotate }
xr_dset = xr.tutorial.open_dataset("air_temperature")
single_time_instant = xr_dset.cm.itime(0)
several_time_instants = xr_dset.cm.itime([0, 100])
several_time_instants = xr_dset.cm.itime(slice(5, 200))
```

or by date:

``` { .python .copy .annotate }
xr_dset = xr.tutorial.open_dataset("air_temperature")
single_time_instant = xr_dset.cm.time("2013-02-10")
several_time_instants = xr_dset.cm.time(["2013-02-10", "2013-02-12"])
several_time_instants = xr_dset.cm.time(slice("2013-02-10", "2013-02-12"))
```

???+ tip
    You can also use [`sel`](/climatrix/api#climatrix.dataset.BaseClimatrixDataset.sel) or [`isel`](/climatrix/api#climatrix.dataset.BaseClimatrixDataset.isel) method with `AxisType.TIME`.


### Sampling data

In [Climatrix](https://jameswalczak.github.io/climatrix/latest) there are following sampling methods implemented:

| Sampling | Description | 
| -------- | ----------- |
| uniform  | data are randomly (uniformly) sampled from the entire spatial domain |
| normal   | data are randomly (following normal distribution) sampled around the defined center point (locaion) |


To sample $10\%$ ($0.1$) of spatial points, use:


``` { .python .copy .annotate }
import xarray as xr
import climatrix as cm

xr_dset = xr.tutorial.open_dataset("air_temperature") # (1)!
dset = xr_dset.cm.itime(0) # (2)!

sparse = dset.sample_uniform(portion=0.1)

sparse.plot(title="Uniform Sampling (10%)")
```

1.  We will use tutorial dataset from `xarray`. To use it, some extra packages might be required.
2.  We select just a first time instant (here, 2013-01-01T00:00)

???+ tip
    If you need exact number of resulting points, use `number` parameter. It is valid also for [`sample_normal`](https://jameswalczak.github.io/climatrix/latest/api/#climatrix.dataset.base.BaseClimatrixDataset.sample_normal)

???+ note
    For sampling method, you can specify NaN-policy (`nan` parameter). There are three options:

    - `ignore` - NaN values will be sampled,
    - `raise`  - error will be raised if any NaN valu will be found
    - `resample` - attempts to return not-NaN values

### Reconstructing

The main functionality of the accessor is to ease data reconstruction. You can reconstruct dense domain from a sparse one, or sparse from another sparse.

``` { .python .copy .annotate }
import xarray as xr
import climatrix as cm

xr_dset = xr.tutorial.open_dataset("air_temperature") 
dset = xr_dset.cm.itime(0)
sparse = dset.sample_uniform(portion=0.1) # (1)!

dense = sparse.reconstruct(dset.domain, method="idw") # (2)! 
dense.plot(title="Reconstructed dataset")
```

1.  First, we need to sample some sparse dataset.
2.  Note, we use domain of `dset` not `xr_dset.cm`. We want to reconstruct to the original domain **after** time subsetting

???+ note
    You can pass extra reconstructor-specific arguments as the last (`recon_kwargs`) argument of the `reconstruct` method. To find definitions of these extra arguments, refer to [Reconstruction](https://jameswalczak.github.io/climatrix/latest/api/#reconstructors) section in [API reference](https://jameswalczak.github.io/climatrix/latest/api/).

### Plotting

To plot dataset (for either dense or sparse domain), just use [`plot`](https://jameswalczak.github.io/climatrix/latest/api/#climatrix.dataset.base.BaseClimatrixDataset.plot) method:


``` { .python .copy .annotate }
dset = xr_dset.cm.itime(0).plot()
```

???+ warning
    At the moment, plotting is enabled **only** for static datasets.
    Remember to select a single time instant before plotting.


