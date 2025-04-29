import xarray as xr

dset = xr.open_dataset("data/era5-land.nc").cm
europe = dset.subset(north=71, south=36, west=-24, east=35).isel_time(0)
sparse = europe.sample(number=10_000, nan_policy="resample")
dense = sparse.reconstruct(
    europe.domain,
    method="siren",
    num_surface_points=5000,
    num_off_surface_points=5000,
    checkpoint="artifacts/siren.ckpt",
)
dense.plot()
