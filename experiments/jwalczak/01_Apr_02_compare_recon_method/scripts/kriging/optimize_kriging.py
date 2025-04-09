"""
This script manages Ordinal Kriging reconstruction and hyper-parameter
optimisation
"""

import os
from functools import partial

import xarray as xr
from bayes_opt import BayesianOptimization

from climatrix import Comparison

TUNING_DSET_PATH = os.path.join(".", "data", "europe_tuning.nc")
POINTS = 1_000
SAMPLING_TYPE = "uniform"
NAN_POLICY = "resample"
SEED = 1

coordinates_type_mapping = {1: "euclidean", 2: "geographic"}
variogram_model_mapping = {
    1: "linear",
    2: "power",
    3: "gaussian",
    4: "spherical",
    5: "exponential",
    6: "holo-effect",
}


def load_data():
    return xr.open_dataset(TUNING_DSET_PATH).cm


def sample_data(dset):
    return dset.sample(
        number=POINTS, kind=SAMPLING_TYPE, nan_policy=NAN_POLICY
    )


def recon(
    source_dset,
    sparse_dset,
    nlags: int,
    anisotropy_scaling: float,
    coordinates_type_code: int,
    variogram_model_code: int,
) -> float:
    coordinates_type_code = int(coordinates_type_code)
    variogram_model_code = int(variogram_model_code)
    recon_dset = sparse_dset.reconstruct(
        source_dset.domain,
        method="ok",
        nlags=int(nlags),
        anisotropy_scaling=float(anisotropy_scaling),
        coordinates_type=coordinates_type_mapping[coordinates_type_code],
        variogram_model=variogram_model_mapping[variogram_model_code],
    )
    metrics = Comparison(recon_dset, source_dset).compute_report()
    # NOTE: minus to force maximizing
    return -metrics["RMSE"]


def find_hyperparameters():
    dset = load_data()
    sparse_dset = sample_data(dset)
    func = partial(recon, source_dset=dset, sparse_dset=sparse_dset)
    hyperparameters_bounds = {
        "nlags": (2, 50),
        "anisotropy_scaling": (1e-5, 5.0),
        "coordinates_type_code": ("1", "2"),
        "variogram_model_code": ("1", "6"),
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
