import xarray as xr

dset = xr.open_dataset("/storage/tul/projects/climatrix/data/era5-land.nc")
europe = (
    dset.cm.subset(north=71, south=36, west=-24, east=35)
    .isel_time(0)
    .plot(title="Europe temperature", cbar_name="Temperature")
)
