from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import ClassVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr
from rich.console import Console
from scipy.spatial import cKDTree
from scipy.stats import kruskal

import climatrix as cm
from climatrix.dataset.base import AxisType, BaseClimatrixDataset
from climatrix.dataset.domain import Domain
from climatrix.decorators.runtime import log_input
from climatrix.exceptions import (
    OperationNotSupportedForDynamicDatasetError,
    ReconstructorConfigurationFailed,
)
from climatrix.optim.hyperparameter import Hyperparameter
from climatrix.reconstruct.base import BaseReconstructor
from climatrix.reconstruct.idw import IDWReconstructor
from climatrix.reconstruct.kriging import OrdinaryKrigingReconstructor
from climatrix.reconstruct.mmgn import MMGNReconstructor

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
console = Console()

PointPerSquareKm = int

SEED = 0
cm.seed_all(SEED)
DSET_PATH = (
    Path(__file__).parent.parent.parent
    / "data"
    / "cds"
    / "e-obs-2011-2024-31-0e-europe.nc"
)
RESULT_PATH = Path(__file__).parent / "results" / "density"
RESULT_PATH.mkdir(parents=True, exist_ok=True)
IDW_HPARAM_FILE = (
    Path(__file__).parent.parent.parent
    / "results"
    / "idw"
    / "hparams_summary.csv"
)
OK_HPARAM_FILE = (
    Path(__file__).parent.parent.parent
    / "results"
    / "ok"
    / "hparams_summary.csv"
)
MMGN_HPARAM_FILE = (
    Path(__file__).parent.parent.parent
    / "results"
    / "inr"
    / "mmgn"
    / "hparams_summary.csv"
)

EARTH_RADIUS = 6371.0


def load_eobs_dataset() -> BaseClimatrixDataset:
    log.info("Loading e-obs dataset from %s...", DSET_PATH)
    ds = xr.open_dataset(DSET_PATH).cm
    log.info("Dataset loaded with dimensions: %s", ds.da.dims)
    return ds


def split_train_test(
    ds: BaseClimatrixDataset, train_portion: float
) -> tuple[BaseClimatrixDataset, BaseClimatrixDataset]:
    log.info(
        "Splitting dataset into train and test sets with train portion %.2f...",
        train_portion,
    )
    ds = ds.isel({cm.AxisType.TIME: -1})

    valid_mask = ~ds.da.isnull().values.squeeze()
    valid_indices = np.argwhere(valid_mask)
    np.random.shuffle(valid_indices)
    lats = ds.domain.latitude.values
    lons = ds.domain.longitude.values

    split = int(len(valid_indices) * train_portion)
    train_idx = valid_indices[:split]
    test_idx = valid_indices[split:]

    var = ds.da.values.squeeze()
    train_vals = var[train_idx[:, 0], train_idx[:, 1]]
    test_vals = var[test_idx[:, 0], test_idx[:, 1]]

    train_ds = (
        cm.Domain.from_axes()
        .lat(latitude=lats[train_idx[:, 0]])
        .lon(longitude=lons[train_idx[:, 1]])
        .sparse()
        .to_xarray(train_vals, name="train_values")
        .cm
    )

    test_ds = (
        cm.Domain.from_axes()
        .lat(latitude=lats[test_idx[:, 0]])
        .lon(longitude=lons[test_idx[:, 1]])
        .sparse()
        .to_xarray(test_vals, name="test_values")
        .cm
    )

    return train_ds, test_ds


def sample_density(
    ds: BaseClimatrixDataset, density: PointPerSquareKm, max_points: int
) -> BaseClimatrixDataset:
    log.info(
        "Sampling dataset to target density of %.2f points per square km...",
        density,
    )
    lats = ds.domain.latitude.values

    d_lat_rad = np.deg2rad(0.1)
    d_lon_rad = np.deg2rad(0.1)

    cell_area = (
        (EARTH_RADIUS**2) * np.cos(np.deg2rad(lats)) * d_lat_rad * d_lon_rad
    )

    total_n = len(lats)
    log.info("Total points in dataset: %d", total_n)

    total_area = np.abs(cell_area).sum()
    log.info("Total area covered: %.2f square km", total_area)

    query_n = min(int(total_area * density), total_n)
    if query_n > max_points:
        log.warning(
            "Requested number of points %d exceeds max_points %d. Using max_points.",
            query_n,
            max_points,
        )
        query_n = max_points
    log.info("Sampling %d points...", query_n)

    probabilities = np.abs(cell_area) / np.abs(cell_area).sum()
    sampled_indices = np.random.choice(
        total_n, size=query_n, replace=False, p=probabilities
    )
    return ds.da.isel(point=sampled_indices).cm


def run_experiment(
    reconstructor: type[BaseReconstructor],
    hparams: dict,
    density: PointPerSquareKm,
    save_fig: bool = False,
):
    ds = load_eobs_dataset()
    train_dset, test_dset = split_train_test(ds, train_portion=0.8)
    sampled_ds = sample_density(
        train_dset, density=density, max_points=train_dset.domain.point.size
    )
    if save_fig:
        ax = sampled_ds.plot(
            title=f"Sampled Training Data (Density: {density:.5f} $pts/km^2$, Points: {sampled_ds.domain.point.size})",
            show=False,
        )
        ax.get_figure().savefig(
            RESULT_PATH
            / f"{reconstructor.__name__}_sampled_data_density_{density:.5f}.pdf",
            bbox_inches="tight",
        )
    plt.close()
    reconstructor_instance = reconstructor(
        dataset=sampled_ds,
        target_domain=test_dset.domain,
        **hparams,
    )
    reconstructed_dset = reconstructor_instance.reconstruct()
    cmp = cm.Comparison(reconstructed_dset, test_dset)
    return cmp.compute_report(), sampled_ds.domain.point.size


def analyze_results(method: str, metrics: dict[float, dict[str, float]]):
    import matplotlib
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt
    import seaborn as sns

    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "legend.title_fontsize": 7,
            "lines.linewidth": 1.2,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.5,
            "figure.dpi": 300,
        }
    )

    median_npoints_in_data_set = 5543
    df = pd.DataFrame(
        [
            {
                "density": density,
                "n_points": metric_dict["n_points"],
                **metric_dict["metrics"],
            }
            for density, metric_dict in metrics.items()
        ]
    )
    metrics_df = df.melt(
        id_vars=["density", "n_points"], var_name="metric", value_name="value"
    )
    metrics_df.metric = [
        metric.strip().replace("R^2", r"$R^2$") for metric in metrics_df.metric
    ]

    fig, ax1 = plt.subplots(figsize=(5, 3))  # single column width ~3.5in
    ax2 = ax1.twinx()

    left_metrics = ["RMSE", "MAE", "$R^2$"]
    right_metrics = ["Max Abs Error"]

    left_df = metrics_df[metrics_df["metric"].isin(left_metrics)]
    right_df = metrics_df[metrics_df["metric"].isin(right_metrics)]

    sns.lineplot(x="density", y="value", hue="metric", data=left_df, ax=ax1)
    ax1.set_xlabel(r"Density ($\mathrm{pts/km^2}$)")
    ax1.set_ylabel(r"Value (RMSE, MAE, $R^2$)")

    sns.lineplot(
        x="density",
        y="value",
        hue="metric",
        data=right_df,
        ax=ax2,
        palette=["red"],
        legend=False,
    )
    ax2.set_ylabel(r"Max Abs Error")

    handles1, labels1 = ax1.get_legend_handles_labels()
    max_err_handle = mlines.Line2D(
        [], [], color="red", linewidth=1.2, label="Max Abs Error"
    )
    median_handle = mlines.Line2D(
        [],
        [],
        color="gray",
        linestyle="--",
        linewidth=1.0,
        alpha=0.7,
        label=f"Median pts ({median_npoints_in_data_set})",
    )
    ax1.legend(
        handles1 + [max_err_handle, median_handle],
        labels1
        + ["Max Abs Error", f"Median pts ({median_npoints_in_data_set})"],
        title="Metric",
        framealpha=0.9,
        edgecolor="0.8",
        loc="best",
    )

    ax2.grid(False)
    ax1.set_axisbelow(True)
    ax2.patch.set_visible(False)

    density_to_npoints = (
        df.drop_duplicates("density")
        .set_index("density")["n_points"]
        .sort_index()
    )
    n_ticks = 6
    indices = np.linspace(0, len(density_to_npoints) - 1, n_ticks, dtype=int)
    sampled = density_to_npoints.iloc[indices]

    density_vals = density_to_npoints.index.values
    npoints_vals = density_to_npoints.values
    median_density = np.interp(
        median_npoints_in_data_set, npoints_vals, density_vals
    )
    y_lims = ax1.get_ylim()
    ax1.vlines(
        x=median_density,
        ymin=y_lims[0],
        ymax=y_lims[1],
        color="gray",
        linestyle="--",
        linewidth=1.0,
        alpha=0.7,
    )

    ax3 = ax1.twiny()
    ax3.xaxis.set_ticks_position("bottom")
    ax3.xaxis.set_label_position("bottom")
    ax3.spines["bottom"].set_position(("outward", 42))
    ax3.set_xlim(ax1.get_xlim())
    ax3.set_xticks(list(sampled.index))
    ax3.set_xticklabels(
        list(sampled.values.astype(int)), rotation=45, ha="right", fontsize=7
    )
    ax3.set_xlabel("Number of Points", fontsize=8)
    ax3.grid(False)
    ax3.patch.set_visible(False)

    fig.tight_layout()
    fig.savefig(
        RESULT_PATH / f"metrics_by_density_{method}.pdf",
        bbox_inches="tight",
        dpi=300,
    )
    plt.show()


def load_common_hparams() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    idw_hparams = pd.read_csv(IDW_HPARAM_FILE)
    idw_common_hparams = {
        "power": idw_hparams["power"].median().item(),
        "k": idw_hparams["k"].mode().item(),
    }
    ok_hparams = pd.read_csv(OK_HPARAM_FILE)
    ok_common_hparams = {
        "nlags": int(ok_hparams["nlags"].median().item()),
        "variogram_model": ok_hparams["variogram_model"].mode().item(),
        "anisotropy_scaling": ok_hparams["anisotropy_scaling"].median().item(),
        "coordinates_type": ok_hparams["coordinates_type"].mode().item(),
    }
    mmgn_hparams = pd.read_csv(MMGN_HPARAM_FILE)
    mmgn_common_hparams = {
        "lr": mmgn_hparams["lr"].median().item(),
        "weight_decay": mmgn_hparams["weight_decay"].median().item(),
        "batch_size": int(mmgn_hparams["batch_size"].median().item()),
        "hidden_dim": int(mmgn_hparams["hidden_dim"].median().item()),
        "latent_dim": int(mmgn_hparams["latent_dim"].median().item()),
        "n_layers": int(mmgn_hparams["n_layers"].median().item()),
        "input_scale": mmgn_hparams["input_scale"].median().item(),
        "alpha": mmgn_hparams["alpha"].median().item(),
    }
    return idw_common_hparams, ok_common_hparams, mmgn_common_hparams


def run_idw(hparams: dict):
    idw_results = {}
    for density in [
        0.0001,
        0.0005,
        0.001,
        0.002,
        0.003,
    ]:  # , 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]:
        log.info(
            "Running experiment with density %.2e points per square km...",
            density,
        )
        metrics, n_points = run_experiment(
            IDWReconstructor, hparams, density=density, save_fig=False
        )
        idw_results[density] = {"metrics": metrics, "n_points": n_points}
    analyze_results(IDWReconstructor.NAME, idw_results)


def run_ok(hparams: dict):
    ok_results = {}
    hparams.update({"backend": "vectorized"})
    for density in [0.0001, 0.0005, 0.001, 0.002, 0.003]:
        if density > 0.001:
            hparams.update({"backend": "loop"})
        log.info(
            "Running experiment with density %.2e points per square km...",
            density,
        )
        metrics, n_points = run_experiment(
            OrdinaryKrigingReconstructor,
            hparams,
            density=density,
            save_fig=False,
        )
        ok_results[density] = {"metrics": metrics, "n_points": n_points}
    analyze_results(OrdinaryKrigingReconstructor.NAME, ok_results)


def run_mmgn(hparams: dict):
    mmgn_results = {}
    hparams.update(device="cpu")
    for density in [0.0001, 0.0005, 0.001, 0.002, 0.003]:
        log.info(
            "Running experiment with density %.2e points per square km...",
            density,
        )
        metrics, n_points = run_experiment(
            MMGNReconstructor, hparams, density=density, save_fig=False
        )
        mmgn_results[density] = {"metrics": metrics, "n_points": n_points}
    analyze_results(MMGNReconstructor.NAME, mmgn_results)


if __name__ == "__main__":
    idw_hparams, ok_hparams, mmgn_hparams = load_common_hparams()

    # ########################
    # IDW Reconstructor
    # ########################
    if os.path.exists(
        RESULT_PATH / f"metrics_by_density_{IDWReconstructor.NAME}.pdf"
    ):
        log.info(
            f"Metrics for {IDWReconstructor.NAME} already exist. Skipping experiment."
        )
    else:
        run_idw(idw_hparams)

    # ########################
    # Ordinary Kriging Reconstructor
    # ########################
    if os.path.exists(
        RESULT_PATH
        / f"metrics_by_density_{OrdinaryKrigingReconstructor.NAME}.pdf"
    ):
        log.info(
            f"Metrics for {OrdinaryKrigingReconstructor.NAME} already exist. Skipping experiment."
        )
    else:
        run_ok(ok_hparams)

    # ########################
    # MMGN Reconstructor
    # ########################
    if os.path.exists(
        RESULT_PATH / f"metrics_by_density_{MMGNReconstructor.NAME}.pdf"
    ):
        log.info(
            f"Metrics for {MMGNReconstructor.NAME} already exist. Skipping experiment."
        )
    else:
        run_mmgn(mmgn_hparams)
