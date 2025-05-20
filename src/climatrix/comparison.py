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
    Class for comparing two dense datasets.

    Attributes
    ----------
    sd : DenseDataset
        The source dataset.
    td : DenseDataset
        The target dataset.
    diff : xarray.DataArray
        The difference between the source and target datasets.

    Parameters
    ----------
    predicted_dataset : DenseDataset
        The source dataset.
    true_dataset : DenseDataset
        The target dataset.
    map_nan_from_source : bool, optional
        If True, the NaN values from the source dataset will be
        mapped to the target dataset. If False, the NaN values
        from the target dataset will be used. Default is None,
        which means `False` for sparse datasets and `True`
        for dense datasets.
    """

    def __init__(
        self,
        predicted_dataset: BaseClimatrixDataset,
        true_dataset: BaseClimatrixDataset,
        map_nan_from_source: bool | None = None,
    ):
        from climatrix.dataset.base import BaseClimatrixDataset

        if not isinstance(
            predicted_dataset, BaseClimatrixDataset
        ) or not isinstance(true_dataset, BaseClimatrixDataset):
            raise TypeError(
                "Both datasets must be BaseClimatrixDataset objects"
            )
        self.predicted_dataset = predicted_dataset
        self.true_dataset = true_dataset
        self._assert_static()
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

    def _assert_static(self):
        if (
            self.predicted_dataset.domain.is_dynamic
            or self.true_dataset.domain.is_dynamic
        ):
            raise NotImplementedError(
                "Comparison between dynamic datasets is not yet implemented"
            )

    def plot_diff(self, ax: Axes | None = None) -> Axes:
        """
        Plot the difference between the source and target datasets.

        Parameters
        ----------
        ax : Axes, optional
            The matplotlib axes on which to plot the difference. If None,
            a new set of axes will be created.

        Returns
        -------
        Axes
            The matplotlib axes containing the plot of the difference.
        """
        if ax is None:
            fig, ax = plt.subplots()

        return self.diff.plot(ax=ax)

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
