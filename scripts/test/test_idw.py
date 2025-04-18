import xarray as xr

import climatrix as cm

dset = xr.open_dataset("data/era5-land.nc")
europe = dset.cm.subset(north=71, south=36, west=-24, east=35).isel_time(0)
sparse = europe.sample(number=1000, kind="uniform", nan_policy="resample")
recon = sparse.reconstruct(europe.domain, method="idw", k=10)
recon.plot()
comp = cm.Comparison(europe, recon)
