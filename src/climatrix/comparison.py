from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

from climatrix.decorators import raise_if_not_installed

if TYPE_CHECKING:
    from climatrix.dataset.base import BaseClimatrixDataset

sns.set_style("darkgrid")

log = logging.getLogger(__name__)


class Comparison:
    """
    Class for comparing two datasets (dense or sparse).

    For sparse domains, uses nearest neighbor matching with optional
    distance thresholds to find corresponding observations.

    Attributes
    ----------
    predicted_dataset : BaseClimatrixDataset
        The predicted/source dataset.
    true_dataset : BaseClimatrixDataset
        The true/target dataset.
    diff : BaseClimatrixDataset
        The difference between the predicted and true datasets.
    distance_threshold : float, optional
        Maximum distance for point correspondence in sparse domains.

    Parameters
    ----------
    predicted_dataset : BaseClimatrixDataset
        The predicted/source dataset.
    true_dataset : BaseClimatrixDataset
        The true/target dataset.
    map_nan_from_source : bool, optional
        If True, the NaN values from the source dataset will be
        mapped to the target dataset. If False, the NaN values
        from the target dataset will be used. Default is None,
        which means `False` for sparse datasets and `True`
        for dense datasets.
    distance_threshold : float, optional
        For sparse domains, maximum distance threshold for considering
        points as corresponding. If None, closest points are always matched.
        Only used when both datasets have sparse domains.
    """

    def __init__(
        self,
        predicted_dataset: BaseClimatrixDataset,
        true_dataset: BaseClimatrixDataset,
        map_nan_from_source: bool | None = None,
        distance_threshold: float | None = None,
    ):
        from climatrix.dataset.base import BaseClimatrixDataset

        if not isinstance(
            predicted_dataset, BaseClimatrixDataset
        ) or not isinstance(true_dataset, BaseClimatrixDataset):
            raise TypeError(
                "Both datasets must be BaseClimatrixDataset objects, "
                f"not {predicted_dataset.__class__.__name__} and "
                f"{true_dataset.__class__.__name__}"
            )
        self.predicted_dataset = predicted_dataset
        self.true_dataset = true_dataset
        self.distance_threshold = distance_threshold
        self._assert_static()
        if predicted_dataset.domain.is_sparse or true_dataset.domain.is_sparse:
            if (
                predicted_dataset.domain.is_sparse
                and true_dataset.domain.is_sparse
            ):
                self.diff = self._compute_sparse_diff()
            else:
                raise ValueError(
                    "Comparison between sparse and dense domains is not supported. "
                    "Both datasets must be either sparse or dense."
                )
        else:
            if map_nan_from_source is None:
                map_nan_from_source = not predicted_dataset.domain.is_sparse
            if map_nan_from_source:
                try:
                    self.predicted_dataset = self.predicted_dataset.mask_nan(
                        self.true_dataset
                    )
                except ValueError as err:
                    log.error(
                        "Error while masking NaN values from source dataset. "
                        "Set `map_nan_from_source` to False to skip this step."
                    )
                    raise ValueError(
                        "Error while masking NaN values from source dataset. "
                        "Set `map_nan_from_source` to False to skip this step."
                    ) from err
            self.diff = self.predicted_dataset - self.true_dataset

    def _compute_sparse_diff(self) -> BaseClimatrixDataset:
        """
        Compute differences between sparse datasets using nearest neighbor matching.

        Returns
        -------
        BaseClimatrixDataset
            A sparse dataset containing differences between matched points.
        """
        from scipy.spatial import cKDTree

        # Get spatial points for both datasets
        pred_points = self.predicted_dataset.domain.get_all_spatial_points()
        true_points = self.true_dataset.domain.get_all_spatial_points()

        # Build KDTree for true dataset points
        tree = cKDTree(true_points)

        # Find nearest neighbors for predicted dataset points
        if self.distance_threshold is not None:
            if self.distance_threshold == 0.0:
                threshold = np.finfo(float).eps
            else:
                threshold = self.distance_threshold
            distances, indices = tree.query(
                pred_points, distance_upper_bound=threshold
            )
            valid_mask = distances < np.inf
        else:
            distances, indices = tree.query(pred_points)
            valid_mask = np.ones(len(distances), dtype=bool)

        pred_values = self.predicted_dataset.da.values
        true_values = self.true_dataset.da.values

        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            log.warning(
                "No valid point correspondences found within distance threshold"
            )
            empty_da = self.predicted_dataset.da.isel(
                {self.predicted_dataset.domain.point.name: []}
            )
            return type(self.predicted_dataset)(empty_da)

        result_values = (pred_values - true_values[indices])[valid_indices]

        result_da = self.predicted_dataset.da.isel(
            {self.predicted_dataset.domain.point.name: valid_indices}
        )
        result_da = result_da.copy()
        result_da.values = result_values

        return type(self.predicted_dataset)(result_da)

    def _assert_static(self):
        if (
            self.predicted_dataset.domain.is_dynamic
            or self.true_dataset.domain.is_dynamic
        ):
            raise NotImplementedError(
                "Comparison between dynamic datasets is not yet implemented"
            )

    def plot_diff(
        self,
        title: str | None = None,
        target: str | os.PathLike | Path | None = None,
        show: bool = False,
        ax: Axes | None = None,
        **kwargs,
    ) -> Axes:
        """
        Plot the difference between the source and target datasets.

        Parameters
        ----------
        title : str, optional
            Title of the plot. If not provided, the name of the dataset
            will be used. If the dataset has no name, "Climatrix Dataset" will be used.
        target : str, os.PathLike, Path, or None, optional
            Path to save the plot. If not provided, the plot
            will not be saved.
        show : bool, optional
            Whether to show the plot. Default is False.
        ax : Axes, optional
            Axes to plot on. If not provided, a new figure and axes
            will be created.
        **kwargs : dict, optional
            Additional keyword arguments to pass to the plotting function.

            - `figsize`: tuple, optional
                Size of the figure. Default is (12, 6).
            - `vmin`: float, optional
                Minimum value for the color scale. Default is None.
            - `vmax`: float, optional
                Maximum value for the color scale. Default is None.
            - `cmap`: str, optional
                Colormap to use for the plot. Default is "seismic".
            - `size`: int, optional
                Size of the points for sparse datasets. Default is 10.

        Returns
        -------
        Axes
            The matplotlib axes containing the plot of the difference.
        """
        return self.diff.plot(
            title=title, target=target, show=show, ax=ax, **kwargs
        )

    def plot_signed_diff_hist(
        self,
        ax: Axes | None = None,
        n_bins: int = 50,
        limits: tuple[float] | None = None,
        label: str | None = None,
        alpha: float = 1.0,
    ) -> Axes:
        """
        Plot the histogram of signed difference between datasets.

        The signed difference is a dataset where positive values
        represent areas where the source dataset is larger than
        the target dataset and negative values represent areas
        where the source dataset is smaller than
        the target dataset.

        Parameters
        ----------
        ax : Axes, optional
            The matplotlib axes on which to plot the histogram. If None,
            a new set of axes will be created.
        n_bins : int, optional
            The number of bins to use in the histogram (default is 50).
        limits : tuple[float], optional
            The limits of values to include in the
            histogram (default is None).

        Returns
        -------
        Axes
            The matplotlib axes containing the plot of the signed difference.
        """
        if ax is None:
            fig, ax = plt.subplots()

        ax.hist(
            self.diff.da.values.flatten(),
            bins=n_bins,
            range=limits,
            label=label,
            alpha=alpha,
        )
        return ax

    def compute_rmse(self) -> float:
        """
        Compute the RMSE between the source and target datasets.

        Returns
        -------
        float
            The RMSE between the source and target datasets.
        """
        nanmean = np.nanmean(np.power(self.diff.da.values, 2.0))
        return np.power(nanmean, 0.5).item()

    def compute_mae(self) -> float:
        """
        Compute the MAE between the source and target datasets.

        Returns
        -------
        float
            The mean absolute error between the source and target datasets.
        """
        return np.nanmean(np.abs(self.diff.da.values)).item()

    def compute_mse(self) -> float:
        """
        Compute the MSE between the source and target datasets.

        Returns
        -------
        float
            The mean squared error between the source and target datasets.
        """
        return np.nanmean(np.power(self.diff.da.values, 2.0)).item()

    def compute(self, metric: str) -> float:
        """
        Compute the specified metric.

        Parameters
        ----------
        metric : str
            The metric to compute. Supported values:
            "mae", "mse", "rmse".

        Returns
        -------
        float
            The computed metric value.

        Raises
        ------
        ValueError
            If the metric is not supported.
        """
        metric = metric.lower().strip()
        if metric == "mae":
            return self.compute_mae()
        elif metric == "mse":
            return self.compute_mse()
        elif metric == "rmse":
            return self.compute_rmse()
        else:
            raise ValueError(
                f"Unsupported metric: {metric}. "
                "Supported metrics: mae, mse, rmse"
            )

    @raise_if_not_installed("sklearn")
    def compute_r2(self):
        """
        Compute the R^2 between the source and target datasets.

        Returns
        -------
        float
            The R^2 between the source and target datasets.
        """
        from sklearn.metrics import r2_score

        sd = self.predicted_dataset.da.values.flatten()
        sd = sd[~np.isnan(sd)]
        td = self.true_dataset.da.values.flatten()
        td = td[~np.isnan(td)]
        return r2_score(sd, td)

    def compute_max_abs_error(self) -> float:
        """
        Compute the maximum absolute error between datasets.

        Returns
        -------
        float
            The maximum absolute error between the source and
            target datasets.
        """
        return np.nanmax(np.abs(self.diff.da.values)).item()

    def compute_report(self) -> dict[str, float]:
        return {
            "RMSE": self.compute_rmse(),
            "MAE": self.compute_mae(),
            "Max Abs Error": self.compute_max_abs_error(),
            "R^2": self.compute_r2(),
        }

    def save_report(self, target_dir: str | os.PathLike | Path) -> None:
        """
        Save a report of the comparison between passed datasets.

        This method will create a directory at the specified path
        and save a report of the comparison between the source and
        target datasets in that directory. The report will include
        plots of the difference and signed difference between the
        datasets, as well as a csv file with metrics such
        as the RMSE, MAE, and maximum absolute error.

        Parameters
        ----------
        target_dir : str | os.PathLike | Path
            The path to the directory where the report should be saved.
        """
        target_dir = Path(target_dir)
        if target_dir.exists():
            warnings.warn(
                "The target directory already exists and will be overwritten."
            )
        target_dir.mkdir(parents=True, exist_ok=True)
        metrics = self.compute_report()
        pd.DataFrame(metrics, index=[0]).to_csv(
            target_dir / "metrics.csv", index=False
        )
        self.plot_diff().get_figure().savefig(target_dir / "diff.svg")
        self.plot_signed_diff_hist().get_figure().savefig(
            target_dir / "signed_diff_hist.svg"
        )
        plt.close("all")
