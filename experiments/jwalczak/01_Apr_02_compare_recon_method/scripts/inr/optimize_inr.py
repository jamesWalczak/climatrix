"""
This script manages IDW reconstruction and hyper-parameter optimisation
"""

import os
from functools import partial

import xarray as xr
from bayes_opt import BayesianOptimization

import climatrix as cm
from climatrix import Comparison

TUNING_DSET_PATH = os.path.join(".", "data", "europe_tuning.nc")
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
    num_surface_points: int,
    num_off_surface_points: int,
    lr: float,
    gradient_clipping_value: float,
    sdf_loss_weight: float,
    inter_loss_weight: float,
    normal_loss_weight: float,
    eikonal_loss_weight: float,
) -> float:
    recon_dset = sparse_dset.reconstruct(
        source_dset.domain,
        method="siren",
        num_surface_points=int(num_surface_points),
        num_off_surface_points=int(num_off_surface_points),
        lr=float(lr),
        num_epochs=MAX_HYPER_PARAMS_EPOCH,
        gradient_clipping_value=float(gradient_clipping_value),
        sdf_loss_weight=float(sdf_loss_weight),
        inter_loss_weight=float(inter_loss_weight),
        normal_loss_weight=float(normal_loss_weight),
        eikonal_loss_weight=float(eikonal_loss_weight),
    )
    metrics = Comparison(recon_dset, source_dset).compute_report()
    # NOTE: minus to force maximizing
    return -metrics["RMSE"]


def find_hyperparameters():
    dset = load_data()
    sparse_dset = sample_data(dset)
    func = partial(recon, source_dset=dset, sparse_dset=sparse_dset)
    hyperparameters_bounds = {
        "num_surface_points": (100, sparse_dset.domain.size),
        "num_off_surface_points": (100, 10 * sparse_dset.domain.size),
        "lr": (1e-5, 5e-1),
        "gradient_clipping_value": (0.0, 1e3),
        "sdf_loss_weight": (0.0, 1e4),
        "inter_loss_weight": (0.0, 1e3),
        "normal_loss_weight": (0.0, 1e3),
        "eikonal_loss_weight": (0.0, 1e2),
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
