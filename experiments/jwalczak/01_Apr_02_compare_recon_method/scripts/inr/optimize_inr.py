"""
This script manages IDW reconstruction and hyper-parameter optimisation
"""

from functools import partial
from pathlib import Path

import xarray as xr
from bayes_opt import BayesianOptimization

from climatrix import Comparison

TUNING_DSET_PATH = (
    Path(__file__).parent.parent.parent / "data" / "ecad_obs_europe_train.nc"
)
POINTS = 1_000
SAMPLING_TYPE = "uniform"
NAN_POLICY = "resample"
SEED = 1
MAX_HYPER_PARAMS_EPOCH = 1_000


def load_data():
    return xr.open_dataset(TUNING_DSET_PATH).cm


def sample_data(dset):
    return dset.sample(
        number=POINTS, kind=SAMPLING_TYPE, nan_policy=NAN_POLICY
    )


def recon(
    source_dset,
    sparse_dset,
    lr: float,
    batch_size: int,
    gradient_clipping_value: float,
    mse_loss_weight: float,
    eikonal_loss_weight: float,
    laplace_loss_weight: float,
) -> float:
    recon_dset = sparse_dset.reconstruct(
        source_dset.domain,
        method="siren",
        batch_size=int(batch_size),
        lr=float(lr),
        num_epochs=MAX_HYPER_PARAMS_EPOCH,
        gradient_clipping_value=float(gradient_clipping_value),
        mse_loss_weight=float(mse_loss_weight),
        eikonal_loss_weight=float(eikonal_loss_weight),
        laplace_loss_weight=float(laplace_loss_weight),
    )
    metrics = Comparison(recon_dset, source_dset).compute_report()
    # NOTE: minus to force maximizing
    return -metrics["RMSE"]


def find_hyperparameters():
    dset = load_data()
    sparse_dset = sample_data(dset)
    func = partial(recon, source_dset=dset, sparse_dset=sparse_dset)
    hyperparameters_bounds = {
        "lr": (1e-5, 1e-2),
        "gradient_clipping_value": (0.0, 1e3),
        "batch_size": (10, 500),
        "mse_loss_weight": (1.0, 1e3),
        "eikonal_loss_weight": (0.0, 1e2),
        "laplace_loss_weight": (0.0, 1e2),
    }

    optimizer = BayesianOptimization(
        f=func,
        pbounds=hyperparameters_bounds,
        random_state=SEED,
    )
    optimizer.maximize(
        init_points=30,
        n_iter=200,
    )
    print(optimizer.max)


if __name__ == "__main__":
    find_hyperparameters()
