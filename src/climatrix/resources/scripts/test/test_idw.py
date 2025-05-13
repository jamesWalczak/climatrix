import xarray as xr

import climatrix as cm

dset = xr.open_dataset("/storage/tul/projects/climatrix/data/era5-land.nc")
europe = (
    dset.cm.to_signed_longitude()
    .subset(north=71, south=36, west=-24, east=35)
    .itime(0)
)
sparse = europe.sample_uniform(number=1000, nan="resample")
sparse.plot()
recon = sparse.reconstruct(europe.domain, method="idw", k=10)
recon.plot()
comp = cm.Comparison(europe, recon)
