"""
This module contains the code for the INR trials.

All hyper-parameters were selected using constrained Bayesian
optimisation.

Bayesian optimisation output (init_points=30, n_iter=200)
{
    'target': np.float64(-2.4601502418518066),
    'params': {
        'batch_size': np.float64(347.51957759714594),
        'gradient_clipping_value': np.float64(180.8058300526181),
        'lr': np.float64(5.404738409598668e-05),
        'mse_loss_weight': np.float64(494.29206838653624),
        'eikonal_loss_weight': np.float64(55.760909614566955),
        'laplace_loss_weight': np.float64(0.15144664737775093),
        }
}
"""

from pathlib import Path

import xarray as xr
from rich.progress import track

import climatrix as cm
from climatrix.dataset.dense import DenseDataset

TRIALS: int = 1  # 30
N_POINTS: int = 1_000

LR: float = 5.405e-5
BATCH_SIZE: int = 347
NUM_EPOCHS: int = 10_000
MSE_LOSS_WEIGHT: float = 494.292
EIKONAL_LOSS_WEIGHT: float = 55.761
LAPLACE_LOSS_WEIGHT: float = 0.151
GRADIENT_CLIPPING_VALUE: float = 180.806

RECON_DATASET_PATH = Path("data/europe_recon.nc")
RESULT_DIR = Path("results/inr")


def load_dataset() -> DenseDataset:
    return xr.open_dataset(RECON_DATASET_PATH).cm


def reconstruct_and_save_report(
    source_dataset: DenseDataset,
    target_dir: Path,
    sampling_policy: str,
    nan_policy: str,
) -> xr.Dataset:
    # if target_dir.exists():
    #     return None
    sparse_dset = source_dataset.sample(
        number=N_POINTS, kind=sampling_policy, nan_policy=nan_policy
    )
    recon_dset = sparse_dset.reconstruct(
        source_dataset.domain,
        method="sinet",
        lr=LR,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        gradient_clipping_value=GRADIENT_CLIPPING_VALUE,
        mse_loss_weight=MSE_LOSS_WEIGHT,
        eikonal_loss_weight=EIKONAL_LOSS_WEIGHT,
        laplace_loss_weight=LAPLACE_LOSS_WEIGHT,
    )
    recon_dset.plot()
    cmp = cm.Comparison(recon_dset, source_dataset)
    cmp.predicted_dataset.plot()
    # cmp.plot_diff()
    # breakpoint()
    cmp.save_report(target_dir)


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
