import xarray as xr

import climatrix as cm  # noqa # pylint: disable=unused-import

xr.open_dataset(
    "/home/jakub/tul/research/climatrix/data/static-era5-land.nc"
).cm
