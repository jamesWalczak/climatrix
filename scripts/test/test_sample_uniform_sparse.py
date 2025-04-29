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
sample = europe.sample_uniform(number=1000, nan="resample")
sample.sample_normal(number=100, nan="resample").plot()
