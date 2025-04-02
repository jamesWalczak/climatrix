"""
This script manages IDW reconstruction and hyper-parameter optimisation
"""

import os
from functools import partial

import xarray as xr
from bayes_opt import BayesianOptimization
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

import climatrix as cm
from climatrix import Comparison

TUNING_DSET_PATH = os.path.join(".", "data", "europe_tuning.nc")
POINTS = 1_000
SAMPLING_TYPE = "uniform"
NAN_POLICY = "resample"
SEED = 1


def load_data():
    return xr.open_dataset(TUNING_DSET_PATH).cm


def sample_data(dset):
    return dset.sample(
        number=POINTS, kind=SAMPLING_TYPE, nan_policy=NAN_POLICY
    )


def recon(source_dset, sparse_dset, k: int, power: float, k_min: int) -> float:
    recon_dset = sparse_dset.reconstruct(
        source_dset.domain,
        method="idw",
        k=int(k),
        power=float(power),
        k_min=int(k_min),
    )
    metrics = Comparison(recon_dset, source_dset).compute_report()
    # NOTE: minus to force maximizing
    return -metrics["RMSE"]


def find_hyperparameters():
    dset = load_data()
    sparse_dset = sample_data(dset)
    func = partial(recon, source_dset=dset, sparse_dset=sparse_dset)
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
