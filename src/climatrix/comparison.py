import os
from pathlib import Path

from matplotlib.axes import Axes

from climatrix.dataset.base import BaseClimatrixDataset


class Comparison:

    def __init__(
        self,
        source_dataset: BaseClimatrixDataset,
        target_dataset: BaseClimatrixDataset,
    ):
        self.sd = source_dataset
        self.td = target_dataset
        # validate type (two dense or two sparse)
        pass

    def plot_diff(self) -> Axes:
        self.diff = self.sd - self.td
        self.diff.plot()

    def plot_signed_diff(self) -> Axes: ...

    def compute_rmse(self): ...

    def compute_mae(self): ...

    def compute_r2(self): ...

    def compute_max_abs_error(self): ...

    def save_report(self, target_dir: str | os.PathLike | Path) -> None:
        target_dir = Path(target_dir)
        if target_dir.exists():
            raise FileExistsError(f"Directory {target_dir} already exists")
        target_dir.mkdir(parents=True)
        # TODO: save plot
        # TODO: save csv with metrics
