#!/usr/bin/env python3
import numpy as np
import xarray as xr

try:
    import climatrix as cm

    print("✓ Climatrix imported successfully")
except ImportError:
    print("✗ Could not import climatrix. Make sure it's installed.")
    exit(1)


def create_dummy_dense_dataset():
    """Create a dummy dense climate dataset with time and spatial dimensions."""
    print("\nCreating dummy dense dataset...")

    lats = np.linspace(-90, 90, 50)
    lons = np.linspace(-180, 180, 100)
    time = np.arange(
        np.datetime64("2020-01-01"),
        np.datetime64("2020-12-31"),
        np.timedelta64(1, "D"),
    )

    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")

    np.random.seed(42)

    base_temp = 15 - 0.7 * np.abs(lat_grid)

    temp_data = np.zeros((len(time), len(lats), len(lons)))
    for i, t in enumerate(time):
        day_of_year = (t - np.datetime64("2020-01-01")) / np.timedelta64(
            1, "D"
        )
        seasonal_factor = 10 * np.cos(2 * np.pi * day_of_year / 365.25)

        seasonal_temp = base_temp + seasonal_factor * np.cos(
            np.radians(lat_grid)
        )

        noise = np.random.normal(0, 2, seasonal_temp.shape)
        temp_data[i] = seasonal_temp + noise

    # Create xarray dataset
    ds = xr.Dataset(
        {"temperature": (["time", "latitude", "longitude"], temp_data)},
        coords={"time": time, "latitude": lats, "longitude": lons},
        attrs={
            "title": "Dummy Global Temperature Dataset",
            "description": "Synthetic daily temperature data for demonstration",
        },
    )

    ds.temperature.attrs = {
        "units": "degrees_C",
        "long_name": "Air Temperature",
        "standard_name": "air_temperature",
    }

    print(f"✓ Dense dataset created: {ds.sizes}")
    return ds.cm


def create_dummy_sparse_dataset():
    """Create a dummy sparse climate dataset with station data."""
    print("\nCreating dummy sparse dataset...")

    np.random.seed(123)
    n_stations = 50

    station_lats = np.random.uniform(-90, 90, n_stations)
    station_lons = np.random.uniform(-180, 180, n_stations)
    station_ids = [f"STATION_{i:03d}" for i in range(n_stations)]

    time = np.arange(
        np.datetime64("2020-01-01"),
        np.datetime64("2020-01-31"),
        np.timedelta64(1, "D"),
    )
    depth = np.arange(0, 20)

    precip_data = np.zeros((len(time), len(depth), n_stations))
    for i, (lat, lon) in enumerate(zip(station_lats, station_lons)):
        base_precip = 5 + 10 * np.exp(-0.02 * lat**2)

        for j in range(len(time)):
            for k in range(len(depth)):
                if np.random.random() < 0.3:  # Rain on 30% of days
                    precip_data[j, k, i] = np.random.exponential(base_precip)
                else:
                    precip_data[j, k, i] = 0

    ds = xr.Dataset(
        {"precipitation": (["time", "depth", "station"], precip_data)},
        coords={
            "time": time,
            "depth": depth,
            "station": station_ids,
            "latitude": ("station", station_lats),
            "longitude": ("station", station_lons),
        },
        attrs={
            "title": "Dummy Station Precipitation Dataset",
            "description": "Synthetic station precipitation data for demonstration",
        },
    )

    ds.precipitation.attrs = {
        "units": "mm/day",
        "long_name": "Daily Precipitation",
        "standard_name": "precipitation_flux",
    }

    print(f"✓ Sparse dataset created: {ds.sizes}")
    return ds.cm


if __name__ == "__main__":
    print("Running interactive plotting demonstration...")
    dset_type = int(
        input("Enter dataset type (1: dense/2: sparse): ").strip().lower()
    )
    if dset_type == 1:
        print("Creating and showing dense dataset plot...")
        dset = create_dummy_dense_dataset()
    elif dset_type == 2:
        print("Creating and showing sparse dataset plot...")
        dset = create_dummy_sparse_dataset()
    else:
        print("Invalid option. Please run again and choose 1 or 2.")
        exit(1)

    plot = cm.plot.Plot(dset)
    plot.show(port=5000)
