import os
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes

from climatrix.dataset.dense import DenseDataset


class Comparison:

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
        return self.diff.da.plot(ax=ax)

    def plot_signed_diff(self) -> Axes:
        # TODO:
        raise NotImplementedError

    def compute_rmse(self) -> float:
        nansum = np.nansum(np.power(self.diff.da.values, 2.0))
        return np.power(nansum, 0.5).item()

    def compute_mae(self) -> float:
        return np.nanmean(np.abs(self.diff.da.values)).item()

    def compute_r2(self):
        # TODO:
        raise NotImplementedError

    def compute_max_abs_error(self) -> float:
        return np.nanmax(np.abs(self.diff.da.values)).item()

    def save_report(self, target_dir: str | os.PathLike | Path) -> None:
        target_dir = Path(target_dir)
        if target_dir.exists():
            raise FileExistsError(f"Directory {target_dir} already exists")
        target_dir.mkdir(parents=True)
        # TODO: save plot
        # TODO: save csv with metrics
