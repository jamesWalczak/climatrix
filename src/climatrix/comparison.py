import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import saeborn as sns
from matplotlib.axes import Axes

from climatrix.dataset.dense import DenseDataset
from src.climatrix.decorators import raise_if_not_installed

sns.set_style("darkgrid")


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
    source_dataset : DenseDataset
        The source dataset.
    target_dataset : DenseDataset
        The target dataset.
    map_nan_from_source : bool, optional
        Whether to map NaN values from the source dataset to the target dataset.
        Default is True.
    """

    def __init__(
        self,
        source_dataset: DenseDataset,
        target_dataset: DenseDataset,
        map_nan_from_source: bool = True,
    ):
        if not isinstance(source_dataset, DenseDataset) or not isinstance(
            target_dataset, DenseDataset
        ):
            raise NotImplementedError(
                "Comparison is currently enabled only for dense datasets."
            )
        self.sd = source_dataset
        self.td = target_dataset
        self._assert_static()
        if map_nan_from_source:
            self.td.mask_nan(self.sd)
        self.diff = self.sd - self.td

    def _assert_static(self):
        if self.sd.is_dynamic or self.td.is_dynamic:
            raise NotImplementedError(
                "Comaprison between dynamic datasets is not yet implemented"
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

        return self.diff.da.plot(ax=ax)

    def plot_signed_diff(self, ax: Axes | None = None) -> Axes:
        # TODO:
        """
        Plot the histogram of signed difference between datasets.

        The signed difference is a dataset where positive values
        represent areas where the source dataset is larger than
        the target dataset and negative values represent areas
        where the source dataset is smaller than
        the target dataset.

        Returns
        -------
        Axes
            The matplotlib axes containing the plot of the signed difference.
        """
        if ax is None:
            fig, ax = plt.subplots()

        ax.hist(self.diff.da.values.flatten())
        raise NotImplementedError

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
        # TODO:
        """
        Compute the R^2 between the source and target datasets.

        Returns
        -------
        float
            The R^2 between the source and target datasets.
        """
        from sklearn.metrics import r2_score

        return r2_score(
            self.sd.da.values.flatten(), self.td.da.values.flatten()
        )

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
            raise FileExistsError(f"Directory {target_dir} already exists")
        target_dir.mkdir(parents=True)
        # TODO: save plot
        # TODO: save csv with metrics
