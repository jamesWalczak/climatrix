import xarray as xr

dset = xr.open_dataset("/storage/tul/projects/climatrix/data/era5-land.nc")
europe = dset.cm.subset(north=71, south=36, west=-24, east=35).isel_time(0)
europe.sample(number=1000, kind="normal").plot()
