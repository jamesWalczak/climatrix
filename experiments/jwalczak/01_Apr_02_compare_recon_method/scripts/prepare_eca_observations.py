import gc
import importlib.resources
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from rich.progress import track

import climatrix as cm

DATA_DIR = importlib.resources.files("climatrix").joinpath(
    "..", "..", "data", "ecad_nonblend"
)
STATIONS_DEF_PATH = os.path.join(DATA_DIR, "sources.txt")
ECA_START_DATE = datetime(1756, 1, 1)
ECA_END_DATE = datetime(2025, 2, 28)
TARGET_FILE = os.path.join(DATA_DIR, "ecad_nonblend.nc")

EXP_DIR = Path(__file__).parent
TUNING_DSET_PATH = EXP_DIR / ".." / "data" / "ecad_obs_europe_tuning.nc"
TUNING_DATE = datetime(2009, 2, 28)
RECON_DSET_PATH = EXP_DIR / ".." / "data" / "ecad_obs_europe_recon.nc"
RECON_DATE = datetime(2010, 2, 28)


def lon_dms_to_decimal(dms_str):
    sign = -1 if dms_str.strip().startswith("-") else 1
    dms_parts = dms_str.strip()[1:].split(":")
    degrees, minutes, seconds = map(float, dms_parts)
    decimal = sign * (degrees + minutes / 60 + seconds / 3600)
    if decimal > 180:
        decimal -= 360
    if not (-180 <= decimal <= 180):
        raise ValueError(f"Invalid longitude: {dms_str}")
    return decimal


def lat_dms_to_decimal(dms_str):
    sign = -1 if dms_str.strip().startswith("-") else 1
    dms_parts = dms_str.strip()[1:].split(":")
    degrees, minutes, seconds = map(float, dms_parts)
    if not (-90 <= degrees <= 90):
        raise ValueError(f"Invalid latitude: {dms_str}")
    return sign * (degrees + minutes / 60 + seconds / 3600)


def load_sources() -> pd.DataFrame:
    """
    Load station metadata handling stations with commas in their names
    """
    # First find the header line
    HEADER_LINES_NBR = 23
    COLUMNS_SPECS = [
        (0, 6),  # SOUID,
        (7, 47),  # SOUNAME,
        (51, 60),  # LAT
        (61, 71),  # LON
        (72, 76),  # HGHT
    ]
    NAMES = [
        "SOUID",
        "SOUNAME",
        "LAT",
        "LON",
        "HGHT",
    ]
    df = pd.read_fwf(
        STATIONS_DEF_PATH,
        skiprows=HEADER_LINES_NBR,
        colspecs=COLUMNS_SPECS,
        names=NAMES,
    )

    df["LAT_degrees"] = df["LAT"].apply(lat_dms_to_decimal)
    df["LON_degrees"] = df["LON"].apply(lon_dms_to_decimal)

    df["HGHT"] = pd.to_numeric(df["HGHT"], errors="coerce")

    return df[["SOUID", "LAT_degrees", "LON_degrees", "HGHT"]]


def load_station_data(souid):
    """Load data for a single station with memory optimization"""
    path = os.path.join(DATA_DIR, f"TG_SOUID{souid}.txt")
    HEADER_LINES_NBR = 19
    COLUMNS_SPECS = [
        (7, 13),  # SOUID,
        (14, 22),  # DATE,
        (23, 28),  # TG
    ]
    NAMES = [
        "SOUID",
        "DATE",
        "TG",
    ]
    df = pd.read_fwf(
        path, skiprows=HEADER_LINES_NBR, colspecs=COLUMNS_SPECS, names=NAMES
    )

    df = df.dropna(subset=["TG"])
    if df.empty:
        return None

    df["TG"] = df["TG"] / 10.0

    df["DATE"] = pd.to_datetime(df["DATE"], format="%Y%m%d")
    df["TG"] = np.where(df["TG"] == -999.9, np.nan, df["TG"])

    df = df.set_index("DATE")
    return df["TG"]


def get_time_range():
    """Determine the full time range without loading all data at once"""
    min_date = pd.Timestamp(ECA_START_DATE)
    max_date = pd.Timestamp(ECA_END_DATE)
    return pd.date_range(start=min_date, end=max_date, freq="D")


def process_in_chunks(metadata_df, time_index):
    """Process stations in chunks to reduce memory usage"""
    num_stations = len(metadata_df)
    ds = xr.Dataset(
        data_vars={
            "mean_temperature": (
                ["valid_time", "point"],
                np.zeros((len(time_index), num_stations), dtype=np.float32)
                * np.nan,
            )
        },
        coords={
            "valid_time": time_index,
            "point": np.arange(num_stations),
            "latitude": ("point", np.zeros(num_stations)),
            "longitude": ("point", np.zeros(num_stations)),
            "height": ("point", np.zeros(num_stations)),
            "souid": ("point", np.zeros(num_stations, dtype=np.int32)),
        },
    )
    ds.latitude.attrs["units"] = "degrees_north"
    ds.longitude.attrs["units"] = "degrees_east"
    ds.height.attrs["units"] = "m"
    ds.mean_temperature.attrs["units"] = "degC"

    for station in track(range(0, num_stations), description="Processing..."):
        row = metadata_df.iloc[station]
        ds.latitude[station] = row["LAT_degrees"]
        ds.longitude[station] = row["LON_degrees"]
        ds.height[station] = row["HGHT"]
        ds.souid[station] = int(row["SOUID"])

        ts = load_station_data(int(row["SOUID"]))
        if ts is not None:
            if ts.index.shape != time_index.shape:
                mask = np.isin(time_index, ts.index, assume_unique=True)
                ds.mean_temperature[mask, station] = ts.values
            else:
                ds.mean_temperature[:, station] = ts.values

        del ts
        if station % 10 == 0:
            gc.collect()

    ds.to_netcdf(TARGET_FILE, mode="w")


def prepare_tuning_dataset():
    xr.open_dataset(TARGET_FILE).sel(valid_time=TUNING_DATE).to_netcdf(
        TUNING_DSET_PATH
    )


def prepare_recon_dataset():
    xr.open_dataset(TARGET_FILE).sel(valid_time=RECON_DATE).to_netcdf(
        RECON_DSET_PATH
    )


if __name__ == "__main__":
    sources = load_sources()
    time_index = get_time_range()
    process_in_chunks(sources, time_index)

    prepare_tuning_dataset()
    prepare_recon_dataset()
