import os

import xarray as xr

import climatrix as cm

TUNING_DSET_PATH = os.path.join(".", "data", "europe_tuning.nc")
RECON_DSET_PATH = os.path.join(".", "data", "europe_recon.nc")

BBOX = {"north": 71, "south": 36, "west": -24, "east": 35}


def prepare_tuning_dataset():
    xr.open_dataset("./data/era5_land_01_03_2025.nc").cm.subset(
        **BBOX
    ).isel_time(0).da.to_netcdf(TUNING_DSET_PATH)


def prepare_recon_dataset():
    xr.open_dataset("./data/era5_land_28_03_2025.nc").cm.subset(
        **BBOX
    ).isel_time(0).da.to_netcdf(RECON_DSET_PATH)


if __name__ == "__main__":
    prepare_tuning_dataset()
    prepare_recon_dataset()
