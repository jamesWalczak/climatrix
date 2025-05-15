import xarray as xr

import climatrix as cm

dset = xr.open_dataset("data.nc")
europe = (
    dset.cm.to_signed_longitude()
    .subset(north=71, south=36, west=-24, east=35)
    .itime(0)
)

sparse = europe.sample_uniform(number=10_000, nan="resample")

sparse.plot()

dense = sparse.reconstruct(
    europe.domain,
    method="siren",
)
dense.plot()

without_nans = dense.mask_nan(europe)
without_nans.plot()

comp = cm.Comparison(dense, europe)

comp.save_report("./results/")
