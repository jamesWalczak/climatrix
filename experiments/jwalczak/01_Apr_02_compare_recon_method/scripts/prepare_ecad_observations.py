import gc
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

URL = "https://knmi-ecad-assets-prd.s3.amazonaws.com/download/ECA_blend_tg.zip"
ZIP_FILE = Path("/tmp/ecad_blend.zip")

DATA_DIR = Path(CLIMATRIX_EXP_DIR) / "data" / "ecad_blend"
STATIONS_DEF_PATH = DATA_DIR / "sources.txt"
TARGET_FILE = DATA_DIR / "ecad_blend.nc"

NBR_SAMPLES: int = 100

np.random.seed(SEED)


def download_file(url, filename):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(filename, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        return False


def prepare_ecad_data():
    train_files = list(DATA_DIR.parent.glob("ecad_obs_europe_train_*.nc"))
    train_files_present = val_files_present = test_files_present = False
    if len(train_files) >= NBR_SAMPLES:
        train_files_present = True
    val_files = list(DATA_DIR.parent.glob("ecad_obs_europe_val_*.nc"))
    if len(val_files) >= NBR_SAMPLES:
        val_files_present = True
    test_files = list(DATA_DIR.parent.glob("ecad_obs_europe_test_*.nc"))
    if len(test_files) >= NBR_SAMPLES:
        test_files_present = True

    if train_files_present and val_files_present and test_files_present:
        print("ECAD data already prepared. Skipping preparation.")
        return

    # If we have source data unzipped but not process, process them
    if (
        (DATA_DIR / "sources.txt").exists()
        and (DATA_DIR / "stations.txt").exists()
        and DATA_DIR.glob("TG_STAID*.txt")
    ):
        print("ECAD data already downloaded and unzipped. Skipping download.")
        sources, min_date, max_date = load_sources()
        time_index = get_time_range(min_date, max_date)
        process_in_chunks(sources, time_index)
        prepare_splits()
        return

    # If we already have the zip file, let us extract it and use
    if ZIP_FILE.exists():
        print(f"Extracting {ZIP_FILE} to {DATA_DIR}")
        import zipfile

        with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
            zip_ref.extractall(DATA_DIR)
        print("Extraction completed.")
        sources, min_date, max_date = load_sources()
        time_index = get_time_range(min_date, max_date)
        process_in_chunks(sources, time_index)
        prepare_splits()
        return

    # If we do not have the zip file, we download it
    print(f"Downloading ECAD data from {URL} to {ZIP_FILE}")
    if not download_file(URL, ZIP_FILE):
        raise RuntimeError(
            f"Failed to download ECAD data from {URL} to {ZIP_FILE}"
        )

    print(
        f"Downloaded ECAD data to {ZIP_FILE}. Now extracting it to {DATA_DIR}"
    )
    import zipfile

    with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)
    print("Extraction completed.")

    # After extraction, we can process the data
    sources, min_date, max_date = load_sources()
    time_index = get_time_range(min_date, max_date)
    process_in_chunks(sources, time_index)
    prepare_splits()


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

    return (
        df[["STATION_ID", "LAT_degrees", "LON_degrees", "HGHT"]],
        min_date,
        max_date,
    )


def load_station_data(station_id):
    """Load data for a single station with memory optimization"""
    path = os.path.join(DATA_DIR, f"TG_STAID{str(station_id).zfill(6)}.txt")
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
        print("Target file exists. Skipping...")
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


def prepare_splits():
    TRAIN_PORTION = 0.6
    VALIDATION_PORTION = 0.2
    full_dset = xr.open_dataset(TARGET_FILE)

    sampled = 0
    while sampled < NBR_SAMPLES:
        date_id = np.random.randint(0, full_dset.valid_time.size - 1, 1).item()
        print(f"Preparing splits for date ID: {date_id}")
        dset = full_dset.isel(valid_time=date_id)
        date = pd.to_datetime(dset.valid_time.values)
        TRAIN_DSET_PATH = (
            CLIMATRIX_EXP_DIR
            / "data"
            / f"ecad_obs_europe_train_{date:%Y%m%d}.nc"
        )
        VALIDATION_DSET_PATH = (
            CLIMATRIX_EXP_DIR
            / "data"
            / f"ecad_obs_europe_val_{date:%Y%m%d}.nc"
        )
        TEST_DSET_PATH = (
            CLIMATRIX_EXP_DIR
            / "data"
            / f"ecad_obs_europe_test_{date:%Y%m%d}.nc"
        )

        dset = dset.dropna(dim="point", how="all")
        if len(dset.point.values) < 500:
            print(
                f"Skipping date ID {date_id} due to insufficient data: {len(dset.point.values)}."
            )
            continue
        idx = np.arange(len(dset["point"]))
        np.random.shuffle(idx)
        train_idx = idx[: int(len(idx) * TRAIN_PORTION)]
        val_idx = idx[
            int(len(idx) * TRAIN_PORTION) : int(
                len(idx) * (TRAIN_PORTION + VALIDATION_PORTION)
            )
        ]
        test_idx = idx[int(len(idx) * (TRAIN_PORTION + VALIDATION_PORTION)) :]
        train_dset = dset.isel(point=train_idx)
        val_dset = dset.isel(point=val_idx)
        test_dset = dset.isel(point=test_idx)
        train_dset.to_netcdf(TRAIN_DSET_PATH)
        val_dset.to_netcdf(VALIDATION_DSET_PATH)
        test_dset.to_netcdf(TEST_DSET_PATH)
        sampled += 1


if __name__ == "__main__":
    prepare_ecad_data()
