"""
This module contains the code for the IDW trials.

All hyper-parameters were selected using constrained Bayesian
optimisation.

Bayesian optimisation output (init_points=30, n_iter=100)
{
    'target (RMSE)': np.float64(-1.4241189958579608),
    'params':
    {
        'k': np.float64(27.40201996616449),
        'k_min': np.float64(17.348586061728497),
        'power': np.float64(2.7965365027773164)
    }
}
"""

from pathlib import Path

import xarray as xr
from rich.progress import track

import climatrix as cm
from climatrix.dataset.dense import DenseDataset

TRIALS: int = 30
N_POINTS: int = 1_000

K: int = 27
POWER: float = 2.79
K_MIN: int = 17

RECON_DATASET_PATH = Path("data/europe_recon.nc")
RESULT_DIR = Path("results/idw")


def load_dataset() -> DenseDataset:
    return xr.open_dataset(RECON_DATASET_PATH).cm


def reconstruct_and_save_report(
    source_dataset: DenseDataset,
    target_dir: Path,
    sampling_policy: str,
    nan_policy: str,
) -> xr.Dataset:
    if target_dir.exists():
        return None
    sparse_dset = source_dataset.sample(
        number=N_POINTS, kind=sampling_policy, nan_policy=nan_policy
    )
    recon_dset = sparse_dset.reconstruct(
        source_dataset.domain,
        method="idw",
        k=K,
        power=POWER,
        k_min=K_MIN,
    )
    cm.Comparison(recon_dset, source_dataset).save_report(target_dir)


def run_experiment_uniform_sampling(source_dataset: DenseDataset):
    for i in track(range(1, TRIALS + 1), description="Reconstructing..."):
        res_dir = RESULT_DIR / "uniform" / f"trial_{i}"
        reconstruct_and_save_report(
            source_dataset, res_dir, "uniform", "resample"
        )


def run_experiment_normal_sampling(source_dataset: DenseDataset):
    for i in track(range(1, TRIALS + 1), description="Reconstructing..."):
        res_dir = RESULT_DIR / "normal" / f"trial_{i}"
        reconstruct_and_save_report(
            source_dataset, res_dir, "normal", "resample"
        )


if __name__ == "__main__":
    dset = load_dataset()
    run_experiment_uniform_sampling(dset)
    # NOTE: normal sampling with "resample" policy has not yet been implemented
    # run_experiment_normal_sampling(dset)
