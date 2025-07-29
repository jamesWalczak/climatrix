import shutil

import matplotlib.pyplot as plt
import xarray as xr

import climatrix as cm

SEED = 0

EUROPE_BOUNDS = {"north": 71, "south": 36, "west": -24, "east": 35}
EUROPE_DOMAIN = cm.Domain.from_lat_lon(
    lat=slice(EUROPE_BOUNDS["south"], EUROPE_BOUNDS["north"], 0.5),
    lon=slice(EUROPE_BOUNDS["west"], EUROPE_BOUNDS["east"], 0.5),
    kind="dense",
)


EUROPE_BOUNDS = {"north": 90, "south": -90, "west": -90, "east": 90}
EUROPE_DOMAIN = cm.Domain.from_lat_lon(
    lat=slice(EUROPE_BOUNDS["south"], EUROPE_BOUNDS["north"], 1),
    lon=slice(EUROPE_BOUNDS["west"], EUROPE_BOUNDS["east"], 1),
    kind="dense",
)
cm.seed_all(SEED)

dset = xr.open_dataset(
    "/storage/tul/projects/climatrix/experiments/jwalczak/01_Apr_02_compare_recon_method/data/ecad_obs_europe_train_19100130.nc"
).cm
val_dset = xr.open_dataset(
    "/storage/tul/projects/climatrix/experiments/jwalczak/01_Apr_02_compare_recon_method/data/ecad_obs_europe_val_19100130.nc"
).cm
dense = dset.reconstruct(
    EUROPE_DOMAIN,
    method="sinet",
    lr=3e-3,
    batch_size=512,
    num_epochs=200,
    num_workers=0,
    device="cuda",
    gradient_clipping_value=None,
    mse_loss_weight=1e0,
    eikonal_loss_weight=2e-1,
    laplace_loss_weight=1e-5,
    patience=None,
    checkpoint="./sinet.ckpt",
    overwrite_checkpoint=True,
    use_elevation=True,
)
dense.plot()
recon_val_dset = dset.reconstruct(
    val_dset.domain,
    method="sinet",
    checkpoint="./sinet.ckpt",
    # use_slope=True,
    use_elevation=True,
)
# # val_dset.plot()
cmp = cm.Comparison(val_dset, recon_val_dset)
cmp.plot_signed_diff_hist()
print(cmp.compute_report())
plt.show()
