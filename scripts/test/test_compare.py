import xarray as xr

dset = xr.open_dataset(
    "/home/jakub/tul/research/climatrix/data/static-era5-land.nc"
)
europe = dset.cm.subset(north=71, south=36, west=-24, east=35).isel_time(0)
sum = europe + 1000
sum.plot()
