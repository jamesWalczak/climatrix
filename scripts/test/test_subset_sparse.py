import xarray as xr

import climatrix as cm

dset = xr.open_dataset(
    "/home/jakub/tul/research/climatrix/data/static-era5-land.nc"
)
europe = dset.cm.itime(0)
sample = (
    europe.sample_uniform(number=100_000, nan="resample")
    .subset(north=71, south=36, west=-24, east=35)
    .plot()
)
