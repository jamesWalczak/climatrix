import xarray as xr

import climatrix as cm

dset = xr.open_dataset(
    "/home/jakub/tul/research/climatrix/data/static-era5-land.nc"
)
europe = (
    dset.cm.to_signed_longitude()
    .subset(north=71, south=36, west=-24, east=35)
    .itime(0)
)
europe.sample(number=1000, kind="uniform", nan_policy="resample").plot()
