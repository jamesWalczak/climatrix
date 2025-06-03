# 🧑‍💻️ Climatrix Command Line Interface (CLI)

???+ warning
    The `climatrix` CLI is currently experimental and offers limited functionality.

The `climatrix` library includes a command-line interface (CLI) to help you:

- ✅ Download datasets from the CDS API


## 📦 Dataset Features
Climatrix simplifies access to the (Copernicus Climate Data Store (CDS))[https://cds.climate.copernicus.eu/] API through an easy-to-use command-line interface.

???+ info
    To access CDS datasets, you must:

    - Register on the CDS website

    - Agree to the license terms

    - Configure climatrix with your CDS API credentials


## ✅ Supported Datasets

Currently, Climatrix supports downloading the following datasets:

1. [ERA5-Land Reanalysis](https://doi.org/10.24381/cds.e2161bac)
2. [E-OBS](https://doi.org/10.24381/cds.151d3ec6)

## 🗂 List Available Datasets
To see the list of available datasets:


```bash
cm dataset list
```

to get the list of available datasets.

## ⚙️ Configure credentials
Before downloading data, configure the related datastore:

???+ note
    At the moment, we support only access to (Copernicus Climate Data Store (CDS))[https://cds.climate.copernicus.eu/]

```bash
cm dataset config cds
```

Follow the prompts to enter your CDS API key and URL (if the default one is not suitable).

## ⬇️ Download a Dataset

To download a dataset, use:

```bash
cm dataset download [dataset-name] [options]
```

Replace [dataset-name] with one of the supported datasets (see (Supported Datasets)[#supported-datasets]).

| Option | Short name | Description|
| ------ | ---------- | ---------- |
| `--year` | `-y` | Year of data to download |
| `--month` | `-m` | Month of data to download|
| `--day`  | `-d` | Day of data to download | 
| `--hour` | `-h` | Hour of data to download |
| `--target` | `-t` | Path to save the downloaded dataset |
| `--variable` | `-v` | Variables to download


<details>
<summary>Download temperature and surface pressure</summary>

```bash
cm dataset download era5-land -y 2024 -m 10 -d 10 -h 15 -v 2m_temperature -v surface_pressure -t ./test.nc
```

</details>