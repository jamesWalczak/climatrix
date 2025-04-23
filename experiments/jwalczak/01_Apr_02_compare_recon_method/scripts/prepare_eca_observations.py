import os
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
from dask import delayed
from dask.base import compute
from rich.progress import track

DATA_DIR = os.path.join("..", "..", "..", "..", "data", "eca_nonblend")
STATIONS_DEF_PATH = os.path.join(DATA_DIR, "sources.txt")
ECA_START_DATE = datetime(1756, 1, 1)
ECA_END_DATE = datetime(2025, 2, 28)
TARGET_FILE = os.path.join(DATA_DIR, "eca_nonblend.nc")


def lon_dms_to_decimal(dms_str):
    sign = -1 if dms_str.strip().startswith("-") else 1
    dms_parts = dms_str.strip()[1:].split(":")
    degrees, minutes, seconds = map(float, dms_parts)
    decimal = sign * (degrees + minutes / 60 + seconds / 3600)
    if decimal > 180:
        decimal -= 360
    if not (-180 <= decimal <= 180):
        raise ValueError(f"Invalid longitude: {dms_str}")
    return sign * (degrees + minutes / 60 + seconds / 3600)


def lat_dms_to_decimal(dms_str):
    sign = -1 if dms_str.strip().startswith("-") else 1
    dms_parts = dms_str.strip()[1:].split(":")
    degrees, minutes, seconds = map(float, dms_parts)
    if not (-90 <= degrees <= 90):
        raise ValueError(f"Invalid latitude: {dms_str}")
    return sign * (degrees + minutes / 60 + seconds / 3600)


def load_sources() -> pd.DataFrame:
    with open(STATIONS_DEF_PATH, encoding="utf-8") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if "SOUID,SOUNAME" in line.strip():
            data_start_line = i
            break
    df = pd.read_csv(
        STATIONS_DEF_PATH,
        skiprows=data_start_line + 1,
        skipinitialspace=True,
        on_bad_lines="skip",
        names=[
            "SOUID",
            "SOUNAME",
            "CN",
            "LAT",
            "LON",
            "HGHT",
            "ELEID",
            "START",
            "STOP",
            "PARID",
            "PARNAME",
        ],
    )
    df["LAT_degrees"] = df["LAT"].apply(lon_dms_to_decimal)
    df["LON_degrees"] = df["LON"].apply(lon_dms_to_decimal)
    return df[["SOUID", "LAT_degrees", "LON_degrees", "HGHT"]]


def load_data(path):
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if "SOUID," in line.strip():
            data_start_line = i
            break
    df = pd.read_csv(
        path,
        skiprows=data_start_line + 1,
        skipinitialspace=True,
        on_bad_lines="skip",
        names=["STAID", "SOUID", "DATE", "TG", "Q_TG"],
    )
    df["TG"] = df["TG"].astype(float) / 10.0
    df["DATE"] = pd.to_datetime(df["DATE"], format="%Y%m%d")
    return df[["DATE", "TG"]]


BBOX = {"north": 71, "south": 36, "west": -24, "east": 35}


def create_dataarray(metadata_df):
    all_times = set()
    station_data = []

    for idx, row in track(metadata_df.iterrows(), total=len(metadata_df)):
        souid = int(row["SOUID"])
        lat = row["LAT_degrees"]
        lon = row["LON_degrees"]
        height = row["HGHT"]

        df = load_data(os.path.join(DATA_DIR, f"TG_SOUID{souid}.txt"))
        df = df.dropna(subset=["TG"])

        all_times.update(df["DATE"].unique())

        station_data.append(
            {
                "souid": souid,
                "lat": lat,
                "lon": lon,
                "height": height,
                "time_series": df.set_index("DATE")["TG"],
            }
        )

    all_times = sorted(all_times)
    time_index = pd.DatetimeIndex(all_times)

    values_array = np.full((len(time_index), len(station_data)), np.nan)

    for i, station in track(enumerate(station_data)):
        ts = station["time_series"]
        aligned_ts = ts.reindex(time_index)
        values_array[:, i] = aligned_ts.values

    lats = [s["lat"] for s in station_data]
    lons = [s["lon"] for s in station_data]
    souids = [s["souid"] for s in station_data]

    data_array = xr.DataArray(
        data=values_array,
        dims=["time", "point"],
        coords={
            "time": time_index,
            "point": np.arange(len(station_data)),
            "latitude": ("point", lats),
            "longitude": ("point", lons),
            "height": ("point", [s["height"] for s in station_data]),
            "souid": ("point", souids),
        },
        name="mean_temperature",
    )

    return data_array


if __name__ == "__main__":
    sources: pd.DataFrame = load_sources()
    da = create_dataarray(sources)
    da.to_netcdf(TARGET_FILE)
