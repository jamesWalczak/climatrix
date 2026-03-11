import gc
import logging
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr
from rich.progress import track

import climatrix as cm

SEED: int = 0
CLIMATRIX_EXP_DIR = os.environ.get("CLIMATRIX_EXP_DIR")
if CLIMATRIX_EXP_DIR is None:
    raise ValueError(
        "CLIMATRIX_EXP_DIR environment variable is not set. "
        "Please set it to the path of your experiment directory."
    )
CLIMATRIX_EXP_DIR = Path(CLIMATRIX_EXP_DIR)

DATA_DIR = CLIMATRIX_EXP_DIR / "data" / "ECA_blend_tg"
RAW_DATA_DIR = DATA_DIR / "raw"
STATIONS_DEF_PATH = RAW_DATA_DIR / "sources.txt"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TARGET_FILE = PROCESSED_DATA_DIR / "ecad_blend.nc"

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_SAMPLES_DATA_DIR = PROCESSED_DATA_DIR / "train_samples"
TRAIN_SAMPLES_DATA_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(SEED)

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
log.addHandler(handler)


def prepare_ecad_data():
    if (
        (RAW_DATA_DIR / "sources.txt").exists()
        and (RAW_DATA_DIR / "stations.txt").exists()
        and RAW_DATA_DIR.glob("TG_STAID*.txt")
    ):
        log.info("Loading and processing ECAD data...")
        sources, min_date, max_date = load_sources()
        time_index = get_time_range(min_date, max_date)
        process_in_chunks(sources, time_index)
    elif TARGET_FILE.exists():
        log.info("Processed data file already exists. Skipping processing.")
    else:
        log.error(
            "ECAD data files not found. Please download and place them in the expected location."
        )
        raise FileNotFoundError(
            "ECAD data files not found in the expected location."
        )
    log.info("ECAD data preparation completed.")
    generate_sample(TARGET_FILE)


def lon_dms_to_decimal(dms_str):
    sign = -1 if dms_str.strip().startswith("-") else 1
    dms_parts = dms_str.strip()[1:].split(":")
    degrees, minutes, seconds = map(float, dms_parts)
    decimal = sign * (degrees + minutes / 60 + seconds / 3600)
    if decimal > 180:
        decimal -= 360
    if not (-180 <= decimal <= 180):
        log.error("Invalid longitude: %s", dms_str)
        raise ValueError(f"Invalid longitude: {dms_str}")
    return decimal


def lat_dms_to_decimal(dms_str):
    sign = -1 if dms_str.strip().startswith("-") else 1
    dms_parts = dms_str.strip()[1:].split(":")
    degrees, minutes, seconds = map(float, dms_parts)
    if not (-90 <= degrees <= 90):
        log.error("Invalid latitude: %s", dms_str)
        raise ValueError(f"Invalid latitude: {dms_str}")
    return sign * (degrees + minutes / 60 + seconds / 3600)


def load_sources() -> pd.DataFrame:
    """
    Load station metadata handling stations with commas in their names
    """
    log.info("Loading station metadata from sources.txt...")
    # First find the header line
    HEADER_LINES_NBR = 24
    COLUMNS_SPECS = [
        (1, 5),  # STATION_ID
        (6, 12),  # SOUID
        (13, 53),  # SOUNAME,
        (57, 66),  # LAT
        (67, 77),  # LON
        (78, 82),  # HGHT
        (88, 96),  # START_DATE
        (97, 105),  # END_DATE
    ]
    NAMES = [
        "STATION_ID",
        "SOUID",
        "SOUNAME",
        "LAT",
        "LON",
        "HGHT",
        "START_DATE",
        "END_DATE",
    ]
    df = pd.read_fwf(
        STATIONS_DEF_PATH,
        skiprows=HEADER_LINES_NBR,
        colspecs=COLUMNS_SPECS,
        names=NAMES,
    )
    df["LAT_degrees"] = df["LAT"].apply(lat_dms_to_decimal)
    df["LON_degrees"] = df["LON"].apply(lon_dms_to_decimal)
    df["START_DATE"] = pd.to_datetime(df["START_DATE"], format="%Y%m%d")
    df["END_DATE"] = pd.to_datetime(df["END_DATE"], format="%Y%m%d")
    min_date = np.min(df["START_DATE"])
    max_date = np.max(df["END_DATE"])

    df["HGHT"] = pd.to_numeric(df["HGHT"], errors="coerce")

    log.info("Station metadata loaded and processed.")
    return (
        df[["STATION_ID", "LAT_degrees", "LON_degrees", "HGHT"]],
        min_date,
        max_date,
    )


def load_station_data(station_id):
    """Load data for a single station with memory optimization"""
    log.info("Loading data for station ID: %d", station_id)
    path = os.path.join(
        RAW_DATA_DIR, f"TG_STAID{str(station_id).zfill(6)}.txt"
    )
    HEADER_LINES_NBR = 21
    COLUMNS_SPECS = [
        (7, 13),  # SOUID,
        (14, 22),  # DATE,
        (23, 28),  # TG
        (29, 34),  # Q_TG
    ]
    NAMES = [
        "SOUID",
        "DATE",
        "TG",
        "Q_TG",
    ]
    df = pd.read_fwf(
        path, skiprows=HEADER_LINES_NBR, colspecs=COLUMNS_SPECS, names=NAMES
    )
    df = df.dropna(subset=["TG"])
    if df.empty:
        return None

    # NOTE: we use just valid values (Q_TG == 0) for temperature
    df = df[df["Q_TG"] == 0]
    if df.empty:
        warnings.warn(f"Station {station_id} has no valid temperature data.")
        return None

    df["TG"] = np.where(df["TG"] == -9999, np.nan, df["TG"])
    df["TG"] = df["TG"] / 10.0

    df["DATE"] = pd.to_datetime(df["DATE"], format="%Y%m%d")

    df = df.set_index("DATE")
    return df["TG"]


def get_time_range(min_date, max_date):
    """Determine the full time range without loading all data at once"""
    return pd.date_range(start=min_date, end=max_date, freq="D")


def process_in_chunks(metadata_df, time_index):
    """Process stations in chunks to reduce memory usage"""
    num_stations = len(metadata_df)
    if TARGET_FILE.exists():
        log.info("Target file exists. Skipping...")
        return
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
            "station_id": ("point", np.zeros(num_stations, dtype=np.int32)),
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
        ds.station_id[station] = int(row["STATION_ID"])

        ts = None
        try:
            ts = load_station_data(int(row["STATION_ID"]))
        except FileNotFoundError:
            log.warning(
                "Data file for station ID %d not found. Filling with NaN.",
                int(row["STATION_ID"]),
            )
            ds.mean_temperature[:, station] = np.nan
            continue

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


def generate_sample(target_file: Path):
    log.info(f"Running sensitivity studies on {target_file}...")
    ds = xr.open_dataset(target_file)
    train_dataset = {
        100: None,
        200: None,
        500: None,
        1_000: None,
        2_000: None,
        5_000: None,
        10_000: None,
        20_000: None,
        30_000: None,
        50_000: None,
        70_000: None,
        100_000: None,
    }
    i = 0
    concat_ds = None
    while True:
        if all(
            train_dataset[size] is not None for size in train_dataset.keys()
        ):
            log.info("All training sample sizes have been filled. Stopping.")
            break
        i -= 1
        last_points = ds.isel(valid_time=i).dropna(dim="point", how="all")
        last_points = last_points.drop_vars("point").assign_coords(
            point=np.arange(last_points.point.size)
        )
        if last_points.point.size == 0:
            log.warning(
                "No valid points found at time index %d. Skipping...", i
            )
            continue
        for train_size in train_dataset.keys():
            if train_dataset[train_size] is not None:
                continue
            if last_points.point.size >= train_size:
                log.info(
                    "Selecting %d points from time index %d for training sample size %d",
                    train_size,
                    i,
                    train_size,
                )
                selected_point = np.random.choice(
                    last_points.point, train_size, replace=False
                )
                train_dataset[train_size] = last_points.sel(
                    point=selected_point
                )
            else:
                log.info(
                    "Not enough points at time index %d for training sample size %d. Found %d points. Adding to merge pool.",
                    i,
                    train_size,
                    last_points.point.size,
                )
                if concat_ds is None:
                    concat_ds = last_points
                else:
                    last_points = last_points.drop_vars("point").assign_coords(
                        point=np.arange(
                            concat_ds.point.size,
                            concat_ds.point.size + last_points.point.size,
                        )
                    )
                    concat_ds = xr.concat(
                        [concat_ds, last_points], dim="point"
                    ).dropna(dim="point", how="all")
                if concat_ds.point.size >= train_size:
                    log.info(
                        "Merged data has enough points for training sample size %d. Selecting from merged data.",
                        train_size,
                    )
                    selected_point = np.random.choice(
                        concat_ds.point, train_size, replace=False
                    )
                    train_dataset[train_size] = concat_ds.sel(
                        point=selected_point
                    )
                else:
                    break
        log.info("Processed time index %d", i)

    for train_size, sample_ds in train_dataset.items():
        if sample_ds is not None:
            sample_path = (
                TRAIN_SAMPLES_DATA_DIR / f"train_sample_{train_size}.nc"
            )
            log.info(
                "Saving training sample of size %d to %s",
                train_size,
                sample_path,
            )
            sample_ds.to_netcdf(sample_path)


if __name__ == "__main__":
    prepare_ecad_data()
