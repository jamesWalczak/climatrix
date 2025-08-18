"""
This module runs experiment of SiNET method

@author: Jakub Walczak, PhD
"""

import csv
import shutil
from pathlib import Path
from typing import Any

import xarray as xr
from rich.console import Console

import climatrix as cm

console = Console()

# Setting up the experiment parameters
NAN_POLICY = "resample"
console.print("[bold green]Using NaN policy: [/bold green]", NAN_POLICY)

SEED = 1
console.print("[bold green]Using seed: [/bold green]", SEED)

DSET_PATH = Path(__file__).parent.parent.parent.parent.joinpath("data")
console.print("[bold green]Using dataset path: [/bold green]", DSET_PATH)

OPTIM_N_ITERS: int = 500
console.print(
    "[bold green]Using iterations for optimization[/bold green]", OPTIM_N_ITERS
)

RESULT_DIR: Path = (
    Path(__file__).parent.parent.parent / "results" / "inr" / "sinet"
)
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
    "lr": (1e-5, 1e-2),
    "num_epochs": (50, 500),
    "gradient_clipping_value": (1e-4, 1e4),
    "batch_size": (32, 1024),
    "mse_loss_weight": (1e-5, 1),
    "eikonal_loss_weight": (0, 1e-2),
    "laplace_loss_weight": (0, 1e-2),
    "patience": (10, 200),
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


def get_all_dataset_idx() -> list[str]:
    return sorted(
        list({path.stem.split("_")[-1] for path in DSET_PATH.glob("*.nc")})
    )


def update_hparams_csv(hparam_path: Path, hparams: dict[str, Any]):
    fieldnames = [
        "dataset_id",
        "lr",
        "num_epochs",
        "gradient_clipping_value",
        "batch_size",
        "mse_loss_weight",
        "eikonal_loss_weight",
        "laplace_loss_weight",
        "patience",
        "opt_loss",
    ]
    if not hparam_path.exists():
        with open(hparam_path, "w") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
    with open(hparam_path, "a") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(hparams)


def update_metric_csv(metrics_path: Path, metrics: dict[str, Any]):
    fieldnames = ["dataset_id", "RMSE", "MAE", "Max Abs Error", "R^2"]
    if not metrics_path.exists():
        with open(metrics_path, "w") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
    with open(metrics_path, "a") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(metrics)


def is_experiment_done(idx: int) -> bool:
    return (PLOT_DIR / f"{idx}_diffs.png").exists()


def run_single_experiment(
    d: str,
    i: int,
    all_samples: int,
    status: Console.status,
    continuous_update: bool = True,
    reconstruct_dense: bool = True,
):
    cm.seed_all(SEED)
    if is_experiment_done(d):
        console.print(
            f"[bold green]Skipping date {d} as it is already done.[/bold green]"
        )
        return
    status.update(
        f"[magenta]Processing date: {d} ({i + 1}/{all_samples})...",
        spinner="bouncingBall",
    )
    train_dset = xr.open_dataset(
        DSET_PATH / f"ecad_obs_europe_train_{d}.nc"
    ).cm
    val_dset = xr.open_dataset(DSET_PATH / f"ecad_obs_europe_val_{d}.nc").cm
    test_dset = xr.open_dataset(DSET_PATH / f"ecad_obs_europe_test_{d}.nc").cm
    status.update(
        f"[magenta]Optimizing hyper-parameters for date: {d} "
        f"({i + 1}/{all_samples})...",
        spinner="bouncingBall",
    )
    finder = cm.optim.HParamFinder(
        "sinet",
        train_dset,
        val_dset,
        metric="mae",
        n_iters=OPTIM_N_ITERS,
        bounds=BOUNDS,
        random_seed=SEED,
    )
    result = finder.optimize()
    console.print("[bold yellow]Optimized parameters:[/bold yellow]")
    console.print(
        "[yellow]Learning rate (lr):[/yellow]", result["best_params"]["lr"]
    )
    console.print(
        "[yellow]Number of epochs:[/yellow]",
        result["best_params"]["num_epochs"],
    )
    console.print(
        "[yellow]Gradient clipping value:[/yellow]",
        result["best_params"]["gradient_clipping_value"],
    )
    console.print(
        "[yellow]Batch size:[/yellow]", result["best_params"]["batch_size"]
    )
    console.print(
        "[yellow]MSE loss weight:[/yellow]",
        result["best_params"]["mse_loss_weight"],
    )
    console.print(
        "[yellow]Eikonal loss weight:[/yellow]",
        result["best_params"]["eikonal_loss_weight"],
    )
    console.print(
        "[yellow]Laplace loss weight:[/yellow]",
        result["best_params"]["laplace_loss_weight"],
    )
    console.print(
        "[yellow]Early stopping patience:[/yellow]",
        result["best_params"]["patience"],
    )
    console.print(
        "[yellow]Use elevation:[/yellow]",
        result["best_params"]["use_elevation"],
    )
    console.print("[yellow]Best loss:[/yellow]", result["best_loss"])
    status.update(
        "[magenta]Reconstructing with optimised parameters...",
        spinner="bouncingBall",
    )
    status.update(
        "[magenta]Concatenating train and validation datasets...",
        spinner="bouncingBall",
    )
    train_val_dset = xr.concat([train_dset.da, val_dset.da], dim="point").cm
    reconstructed_dset = train_val_dset.reconstruct(
        test_dset.domain,
        method="sinet",
        lr=result["best_params"]["lr"],
        num_epochs=result["best_params"]["num_epochs"],
        batch_size=result["best_params"]["batch_size"],
        num_workers=0,
        device="cuda",
        gradient_clipping_value=result["best_params"][
            "gradient_clipping_value"
        ],
        mse_loss_weight=result["best_params"]["mse_loss_weight"],
        eikonal_loss_weight=result["best_params"]["eikonal_loss_weight"],
        laplace_loss_weight=result["best_params"]["laplace_loss_weight"],
        patience=result["best_params"]["patience"],
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
    if reconstruct_dense:
        reconstructed_dense = train_val_dset.reconstruct(
            EUROPE_DOMAIN,
            method="sinet",
            lr=result["best_params"]["lr"],
            num_epochs=result["best_params"]["num_epochs"],
            batch_size=result["best_params"]["batch_size"],
            num_workers=0,
            device="cuda",
            gradient_clipping_value=result["best_params"][
                "gradient_clipping_value"
            ],
            mse_loss_weight=result["best_params"]["mse_loss_weight"],
            eikonal_loss_weight=result["best_params"]["eikonal_loss_weight"],
            laplace_loss_weight=result["best_params"]["laplace_loss_weight"],
            patience=result["best_params"]["patience"],
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
        "[magenta]Saving test dset to " f"{PLOT_DIR} / {d}_test.png...",
        spinner="bouncingBall",
    )
    test_dset.plot(show=False).get_figure().savefig(PLOT_DIR / f"{d}_test.png")
    status.update("[magenta]Evaluating...", spinner="bouncingBall")
    cmp = cm.Comparison(reconstructed_dset, test_dset)
    cmp.diff.plot(show=False).get_figure().savefig(PLOT_DIR / f"{d}_diffs.png")
    cmp.plot_signed_diff_hist().get_figure().savefig(
        PLOT_DIR / f"{d}_hist.png"
    )
    metrics = cmp.compute_report()
    metrics["dataset_id"] = d
    hyperparams = {
        "dataset_id": d,
        "lr": result["best_params"]["lr"],
        "num_epochs": result["best_params"]["num_epochs"],
        "gradient_clipping_value": result["best_params"][
            "gradient_clipping_value"
        ],
        "batch_size": result["best_params"]["batch_size"],
        "mse_loss_weight": result["best_params"]["mse_loss_weight"],
        "eikonal_loss_weight": result["best_params"]["eikonal_loss_weight"],
        "laplace_loss_weight": result["best_params"]["laplace_loss_weight"],
        "patience": result["best_params"]["patience"],
        "opt_loss": result["best_score"],
    }
    if continuous_update:
        console.print("[bold green]Updating metrics file...[/bold green]")
        update_metric_csv(METRICS_PATH, metrics)

        console.print(
            "[bold green]Updating hyperparameters summary...[/bold green]"
        )
        update_hparams_csv(HYPERPARAMETERS_SUMMARY_PATH, hyperparams)

    return (metrics, hyperparams)


def run_all_experiments_sequentially():
    dset_idx = get_all_dataset_idx()
    with console.status("[magenta]Preparing experiment...") as status:
        for i, d in enumerate(dset_idx):
            if is_experiment_done(d):
                console.print(
                    f"[bold green]Skipping date {d} as it is already done.[/bold green]"
                )
                continue
            run_single_experiment(
                d,
                i,
                len(dset_idx),
                status,
                continuous_update=True,
                reconstruct_dense=True,
            )


if __name__ == "__main__":
    # clear_result_dir()
    create_result_dir()
    run_all_experiments_sequentially()
