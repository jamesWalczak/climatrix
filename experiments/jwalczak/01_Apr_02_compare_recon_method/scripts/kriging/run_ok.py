"""
This module runs experiment of IDW method

@author: Jakub Walczak, PhD
"""

import shutil
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Callable

import pandas as pd
import xarray as xr
from bayes_opt import BayesianOptimization
from rich.console import Console

import climatrix as cm

console = Console()

INF_LOSS = -1e4
coordinates_type_mapping = {1: "euclidean", 2: "geographic"}
variogram_model_mapping = {
    1: "linear",
    2: "power",
    3: "gaussian",
    4: "spherical",
    5: "exponential",
    6: "holo-effect",
}

# Setting up the experiment parameters
NAN_POLICY = "resample"
console.print("[bold green]Using NaN policy: [/bold green]", NAN_POLICY)

SEED = 1
cm.seed_all(SEED)
console.print("[bold green]Using seed: [/bold green]", SEED)

DSET_PATH = Path(__file__).parent.parent.parent.joinpath("data")
console.print("[bold green]Using dataset path: [/bold green]", DSET_PATH)

OPTIM_INIT_POINTS: int = 50
console.print(
    "[bold green]Using nbr initial points for optimization: [/bold green]",
    OPTIM_INIT_POINTS,
)

OPTIM_N_ITERS: int = 100
console.print(
    "[bold green]Using iterations for optimization[/bold green]", OPTIM_N_ITERS
)

RESULT_DIR: Path = Path(__file__).parent.parent.parent / "results" / "ok"
PLOT_DIR: Path = RESULT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
console.print("[bold green]Plots will be saved to: [/bold green]", PLOT_DIR)

METRICS_PATH: Path = RESULT_DIR / "metrics.csv"
console.print(
    "[bold green]Metrics will be saved to: [/bold green]", METRICS_PATH
)

HYPERPARAMETERS_SUMMARY_PATH: Path = RESULT_DIR / "hparams_summary.csv"
console.print(
    "[bold green]Hyperparameters summary will be saved to: [/bold green]",
    HYPERPARAMETERS_SUMMARY_PATH,
)

BOUNDS = {
    "nlags": (2, 50),
    "anisotropy_scaling": (1e-5, 5.0),
    "coordinates_type_code": ("1", "2"),
    "variogram_model_code": ("1", "6"),
}
console.print("[bold green]Hyperparameter bounds: [/bold green]", BOUNDS)

EUROPE_BOUNDS = {"north": 71, "south": 36, "west": -24, "east": 35}
EUROPE_DOMAIN = cm.Domain.from_lat_lon(
    lat=slice(EUROPE_BOUNDS["south"], EUROPE_BOUNDS["north"], 0.1),
    lon=slice(EUROPE_BOUNDS["west"], EUROPE_BOUNDS["east"], 0.1),
    kind="dense",
)


def clear_result_dir():
    console.print(
        "[bold red]Clearing result directory for this experiment...[/bold red]"
    )
    shutil.rmtree(RESULT_DIR, ignore_errors=True)


def create_result_dir():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)


def compute_criterion(
    train_dset: cm.BaseClimatrixDataset,
    val_dset: cm.BaseClimatrixDataset,
    **hparams,
) -> float:
    coordinates_type_code = int(hparams["coordinates_type_code"])
    variogram_model_code = int(hparams["variogram_model_code"])
    nlags = int(hparams["nlags"])
    anisotropy_scaling = float(hparams["anisotropy_scaling"])
    recon_dset = train_dset.reconstruct(
        val_dset.domain,
        method="ok",
        nlags=int(nlags),
        anisotropy_scaling=float(anisotropy_scaling),
        coordinates_type=coordinates_type_mapping[coordinates_type_code],
        variogram_model=variogram_model_mapping[variogram_model_code],
        backend="vectorized",
    )
    metrics = cm.Comparison(
        recon_dset, val_dset, map_nan_from_source=False
    ).compute_report()
    # NOTE: minus to force maximizing
    return -metrics["MAE"]


def find_hyperparameters(
    train_dset: cm.BaseClimatrixDataset,
    val_dset: cm.BaseClimatrixDataset,
    func: Callable[
        [cm.BaseClimatrixDataset, cm.BaseClimatrixDataset, dict], float
    ],
    bounds: dict[str, tuple],
    n_init_points: int = 30,
    n_iter: int = 200,
    seed: int = 0,
    verbose: int = 2,
) -> tuple[float, dict[str, float]]:
    """
    Find hyperparameters using Bayesian Optimization.

    Parameters
    ----------
    train_dset : cm.BaseClimatrixDataset
        Training dataset.
    val_dset : cm.BaseClimatrixDataset
        Validation dataset.
    func : Callable
        Function to optimize.
        It should take two datasets and a dictionary of hyperparameters,
        and return a float score.
    bounds : dict[str, tuple]
        Dictionary of hyperparameter bounds.
        Keys are hyperparameter names, values are tuples (min, max).
    n_init_points : int, optional
        Number of initial random points to sample, by default 30.
    n_iter : int, optional
        Number of iterations for optimization, by default 200.
    seed : int, optional
        Random seed for reproducibility, by default 0.
    verbose : int, optional
        Verbosity level of the optimizer, by default 2.

    Returns
    -------
    tuple[float, dict[str, float]]
        Best score and best hyperparameters found.
    """
    func = partial(func, train_dset=train_dset, val_dset=val_dset)
    optimizer = BayesianOptimization(
        f=func, pbounds=bounds, random_state=seed, verbose=verbose
    )
    optimizer.maximize(
        init_points=n_init_points,
        n_iter=n_iter,
    )
    return optimizer.max["target"], (
        int(optimizer.max["params"]["nlags"]),
        float(optimizer.max["params"]["anisotropy_scaling"]),
        coordinates_type_mapping[
            int(optimizer.max["params"]["coordinates_type_code"])
        ],
        variogram_model_mapping[
            int(optimizer.max["params"]["variogram_model_code"])
        ],
    )


def get_all_dataset_idx() -> list[str]:
    return list({path.stem.split("_")[-1] for path in DSET_PATH.glob("*.nc")})


def run_experiment():
    dset_idx = get_all_dataset_idx()
    with console.status("[magenta]Preparing experiment...") as status:
        all_metrics = {}
        hyperparams = defaultdict(list)
        for i, d in enumerate(dset_idx):
            status.update(
                f"[magenta]Processing date: {d} ({i + 1}/{len(dset_idx)})...",
                spinner="bouncingBall",
            )
            train_dset = xr.open_dataset(
                DSET_PATH / f"ecad_obs_europe_train_{d}.nc"
            ).cm
            val_dset = xr.open_dataset(
                DSET_PATH / f"ecad_obs_europe_val_{d}.nc"
            ).cm
            test_dset = xr.open_dataset(
                DSET_PATH / f"ecad_obs_europe_test_{d}.nc"
            ).cm
            status.update(
                f"[magenta]Optimizing hyper-parameters for date: {d} "
                f"({i + 1}/{len(dset_idx)})...",
                spinner="bouncingBall",
            )
            best_loss, (
                nlags,
                anisotroty_scaling,
                coordinates_type,
                variogram_model,
            ) = find_hyperparameters(
                train_dset,
                val_dset,
                compute_criterion,
                BOUNDS,
                n_init_points=OPTIM_INIT_POINTS,
                n_iter=OPTIM_N_ITERS,
                seed=SEED,
                verbose=2,
            )
            console.print("[bold yellow]Optimized parameters:[/bold yellow]")
            console.print("[yellow]Number of lags:[/yellow]", nlags)
            console.print(
                "[yellow]Anisotropy scaling factor:[/yellow]",
                anisotroty_scaling,
            )
            console.print(
                "[yellow]Coordinates type:[/yellow]",
                coordinates_type,
            )
            console.print(
                "[yellow]Variogram model:[/yellow]",
                variogram_model,
            )
            status.update(
                "[magenta]Reconstructing with optimised parameters...",
                spinner="bouncingBall",
            )
            status.update(
                "[magenta]Concatenating train and validation datasets...",
                spinner="bouncingBall",
            )
            train_val_dset = xr.concat(
                [train_dset.da, val_dset.da], dim="point"
            ).cm
            reconstructed_dset = train_val_dset.reconstruct(
                test_dset.domain,
                method="ok",
                nlags=nlags,
                anisotropy_scaling=anisotroty_scaling,
                coordinates_type=coordinates_type,
                variogram_model=variogram_model,
                backend="vectorized",
            )
            status.update(
                "[magenta]Saving reconstructed dset to "
                f"{PLOT_DIR}/{d}_reconstructed.png...",
                spinner="bouncingBall",
            )
            reconstructed_dset.plot(show=False).get_figure().savefig(
                PLOT_DIR / f"{d}_reconstructed.png"
            )

            status.update(
                "[magenta]Reconstructing to dense Europe domain...",
                spinner="bouncingBall",
            )
            reconstructed_dense = reconstructed_dset.reconstruct(
                EUROPE_DOMAIN,
                method="ok",
                nlags=nlags,
                anisotropy_scaling=anisotroty_scaling,
                coordinates_type=coordinates_type,
                variogram_model=variogram_model,
                backend="vectorized",
            )
            status.update(
                "[magenta]Saving reconstructed dense dset to "
                f"{PLOT_DIR}/{d}_reconstructed_dense.png...",
                spinner="bouncingBall",
            )
            reconstructed_dense.plot(show=False).get_figure().savefig(
                PLOT_DIR / f"{d}_reconstructed_dense.png"
            )
            status.update(
                "[magenta]Saving test dset to "
                f"{PLOT_DIR} / {d}_test.png...",
                spinner="bouncingBall",
            )
            test_dset.plot(show=False).get_figure().savefig(
                PLOT_DIR / f"{d}_test.png"
            )
            status.update("[magenta]Evaluating...", spinner="bouncingBall")
            cmp = cm.Comparison(reconstructed_dset, test_dset)
            cmp.diff.plot(show=False).get_figure().savefig(
                PLOT_DIR / f"{d}_diffs.png"
            )
            cmp.plot_signed_diff_hist().get_figure().savefig(
                PLOT_DIR / f"{d}_hist.png"
            )
            metrics = cmp.compute_report()
            all_metrics[d] = metrics

            hyperparams["nlags"].append(nlags)
            hyperparams["anisotropy_scaling"].append(anisotroty_scaling)
            hyperparams["coordinates_type"].append(coordinates_type)
            hyperparams["variogram_model"].append(variogram_model)
            hyperparams["opt_loss"].append(best_loss)
        status.update(
            "[magenta]Saving quality metrics...", spinner="bouncingBall"
        )
        pd.DataFrame(all_metrics).transpose().to_csv(METRICS_PATH)
        status.update(
            "[magenta]Saving hyperparameters summary...",
            spinner="bouncingBall",
        )
        pd.DataFrame(hyperparams).to_csv(HYPERPARAMETERS_SUMMARY_PATH)


if __name__ == "__main__":
    clear_result_dir()
    create_result_dir()
    run_experiment()
