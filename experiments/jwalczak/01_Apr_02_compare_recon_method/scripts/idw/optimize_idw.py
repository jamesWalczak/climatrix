"""
This script manages IDW reconstruction and hyper-parameter optimisation
"""

import os
from functools import partial
from pathlib import Path

import xarray as xr
from bayes_opt import BayesianOptimization

import climatrix as cm

TRAIN_DSET_PATH = Path(__file__).parent.parent.parent.joinpath(
    "data", "ecad_obs_europe_train.nc"
)
VALIDATION_DSET_PATH = Path(__file__).parent.parent.parent.joinpath(
    "data", "ecad_obs_europe_val.nc"
)
NAN_POLICY = "resample"
SEED = 1

cm.seed_all(SEED)


def load_data() -> tuple:
    return (
        xr.open_dataset(TRAIN_DSET_PATH).cm,
        xr.open_dataset(VALIDATION_DSET_PATH).cm,
    )


def recon(train_dset, val_dset, k: int, power: float, k_min: int) -> float:
    if k_min > k:
        return -100
    recon_dset = train_dset.reconstruct(
        val_dset.domain,
        method="idw",
        k=int(k),
        power=float(power),
        k_min=int(k_min),
    )
    metrics = cm.Comparison(recon_dset, val_dset).compute_report()
    # NOTE: minus to force maximizing
    return -metrics["RMSE"]


def find_hyperparameters():
    train_dset, val_dset = load_data()
    func = partial(recon, train_dset=train_dset, val_dset=val_dset)
    hyperparameters_bounds = {
        "k": (1, 50),
        "power": (-2.0, 5.0),
        "k_min": (1, 40),
    }

    optimizer = BayesianOptimization(
        f=func,
        pbounds=hyperparameters_bounds,
        random_state=SEED,
    )
    optimizer.maximize(
        init_points=30,
        n_iter=100,
    )
    print(optimizer.max)


if __name__ == "__main__":
    find_hyperparameters()
