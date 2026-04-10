from enum import StrEnum

from climatrix.types import Km


class DatasetType(StrEnum):
    ERA5_LAND = "era5-land"
    E_OBS = "e-obs"


EARTH_RADIUS: Km = 6371.0
