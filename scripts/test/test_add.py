import xarray as xr

import climatrix as cm  # noqa # pylint: disable=unused-import

dset = xr.open_dataset(
    "/home/jakub/tul/research/climatrix/data/static-era5-land.nc"
)
europe = dset.cm.to_signed_longitude().subset(
    north=71, south=36, west=-24, east=35
)
sum = europe + 1000
sum.plot()
