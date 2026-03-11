from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
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

log = logging.getLogger(__name__)
console = Console()

SEED = 0
DSET_PATH = Path(__file__).parent.parent.parent / "data"
METRICS_PATH = Path(__file__).parent / "results" / "metric_type"
METRICS_PATH.mkdir(parents=True, exist_ok=True)
HPARAM_FILE = (
    Path(__file__).parent.parent.parent
    / "results"
    / "idw"
    / "hparams_summary.csv"
)


class IDWSurrogateReconstructor(IDWReconstructor):
    NAME: ClassVar[str] = "s_idw"

    @log_input(log, level=logging.DEBUG)
    def __init__(
        self,
        dataset: BaseClimatrixDataset,
        target_domain: Domain,
        p: float = 2.0,
        power: float | None = None,
        k: int | None = None,
        k_min: int | None = None,
    ):
        super().__init__(dataset, target_domain, power, k, k_min)
        self.p = p

    def reconstruct(self) -> BaseClimatrixDataset:
        values = self.dataset.da.values.flatten().squeeze()

        log.debug("Building KDtree for efficient nearest neighbor queries...")
        spatial_points = self.dataset.domain.get_all_spatial_points()
        if (
            not isinstance(spatial_points, np.ndarray)
            or spatial_points.ndim != 2
            or spatial_points.shape[1] != 2
        ):
            log.error(
                "Expected a 2D NumPy array with shape (N, 2) from "
                "get_all_spatial_points(), but got %s with shape %s.",
                type(spatial_points),
                getattr(spatial_points, "shape", None),
            )
            raise ValueError(
                "Expected a 2D NumPy array with shape (N, 2) from "
                f"get_all_spatial_points(), but got {type(spatial_points)} "
                f"with shape {getattr(spatial_points, 'shape', None)}."
            )
        kdtree = cKDTree(spatial_points)
        query_points = self.target_domain.get_all_spatial_points()
        log.debug("Querying %d nearest neighbors...", self.k)
        dists, idxs = kdtree.query(
            query_points, k=self.k, workers=-1, p=self.p
        )

        if self.k == 1:
            idxs = idxs[..., np.newaxis]
            dists = dists[..., np.newaxis]
        dists = np.maximum(dists, 1e-10)
        weights = 1.0 / np.power(dists, self.power)
        weights /= np.nansum(weights, axis=1, keepdims=True)

        knn_data = values[idxs]
        valid_mask = np.isfinite(knn_data)
        weights[~valid_mask] = 0.0
        # NOTE: weight_sum should be 1 as it was normalized in 165th line
        weights_sum = np.nansum(weights, axis=1).squeeze()
        interp_vals = np.divide(
            np.nansum(knn_data * weights, axis=1),
            weights_sum,
            where=weights_sum != 0,
            out=None,
        )

        log.debug("Invalidating points with insufficient neighbors...")
        valid_neighbor_counts = np.isfinite(knn_data).sum(axis=1)
        interp_vals[valid_neighbor_counts < self.k_min] = np.nan

        log.debug("Reconstruction completed.")
        return BaseClimatrixDataset(
            self.target_domain.to_xarray(interp_vals, self.dataset.da.name)
        )

    @property
    def num_params(self) -> int:
        """
        Get the number of hyperparameters for the IDW reconstructor.

        For the IDW, the number of parameters of the method corresponds
        to the number of points in the dataset.

        Returns
        -------
        int
            The number of parameters
        """
        return self.dataset.domain.get_all_spatial_points().shape[0]


def get_all_dataset_idx() -> list[str]:
    return sorted(
        list({path.stem.split("_")[-1] for path in DSET_PATH.glob("*.nc")})
    )


def run_experiment(p: float, dataset_id: int | None = None):
    if (METRICS_PATH / f"metrics_p_{p}.csv").exists():
        log.info(f"Metrics for p={p} already exist. Skipping experiment.")
        return
    dset_idx = get_all_dataset_idx()
    hparams = pd.read_csv(HPARAM_FILE).set_index("dataset_id")
    if dataset_id is not None:
        dset_idx = [dset_idx[dataset_id]]
    with console.status(
        f"[magenta]Preparing experiment for p={p}..."
    ) as status:
        all_metrics = []
        for i, d in enumerate(dset_idx):
            cm.seed_all(SEED)
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
                f"[magenta]Optimizing hyper-parameters for date: {d}"
                f" ({i + 1}/{len(dset_idx)})...",
                spinner="bouncingBall",
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
            reconstructed_dset = IDWSurrogateReconstructor(
                train_val_dset,
                test_dset.domain,
                p=p,
                power=hparams.loc[int(d), "power"].item(),
                k=hparams.loc[int(d), "k"].item(),
                k_min=1,
            ).reconstruct()
            status.update("[magenta]Evaluating...", spinner="bouncingBall")
            cmp = cm.Comparison(reconstructed_dset, test_dset)
            metrics: dict[str, Any] = cmp.compute_report()
            metrics["dataset_id"] = d
            metrics["p"] = p
            all_metrics.append(metrics)

        status.update(
            "[magenta]Saving quality metrics...", spinner="bouncingBall"
        )
        pd.DataFrame(all_metrics).to_csv(
            METRICS_PATH / f"metrics_p_{p}.csv", index=False
        )


def analyze_results():
    import matplotlib.pyplot as plt
    import seaborn as sns

    all_metrics = []
    for metric_file in METRICS_PATH.glob("metrics_p_*.csv"):
        df = pd.read_csv(metric_file)
        all_metrics.append(df)
    metrics_df = pd.concat(all_metrics, ignore_index=True)
    metrics_df = metrics_df.melt(
        id_vars=["dataset_id", "p"], var_name="metric", value_name="value"
    )

    for metric_name, group_df in metrics_df.groupby("metric"):
        groups = [
            group["value"].dropna() for _, group in group_df.groupby("p")
        ]
        stat, p_value = kruskal(*groups)
        print(
            f"Kruskal-Wallis test for {metric_name}: H={stat:.3f}, p={p_value:.3e}"
        )

    plt.figure(figsize=(10, 6))
    sns.boxplot(x="p", y="value", hue="metric", data=metrics_df)
    plt.title("RMSE of IDW Reconstructor with Different p Values")
    plt.xlabel("p (Minkowski distance parameter)")
    plt.ylabel("RMSE")
    plt.savefig(METRICS_PATH / "rmse_by_p.pdf")
    plt.show()


if __name__ == "__main__":
    for p in [1.0, 1.5, 2.0, 3.0, 5.0, 10.0, float("inf")]:
        run_experiment(p)
    analyze_results()
