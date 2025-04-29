import xarray as xr

import climatrix as cm

dset = xr.open_dataset("/storage/tul/projects/climatrix/data/era5-land.nc")
europe = dset.cm.subset(north=71, south=36, west=-24, east=35)
europe.plot()
