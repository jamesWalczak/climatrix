from pathlib import Path

import xarray as xr
from rich.console import Console
from rich.table import Table

exp_dir = Path(__file__).parent.parent
exp_data_dir = exp_dir / "data"


def analyse_split(split: str = "train") -> None:
    """
    Analyse the split data and print statistics.
    """
    split_data_path = list(exp_data_dir.glob(f"ecad_obs_europe_{split}_*"))
    print(f"Found {len(split_data_path)} files for split '{split}'.")
    data = xr.open_mfdataset(
        split_data_path, concat_dim="valid_time", combine="nested"
    )

    console = Console()
    table = Table(title=f"{split.capitalize()} Data Statistics")

    table.add_column("Variable", justify="left")
    table.add_column("Mean", justify="right")
    table.add_column("Std Dev", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")

    for var in data.data_vars:
        mean = data[var].mean().compute().item()
        std_dev = data[var].std().compute().item()
        min_val = data[var].min().compute().item()
        max_val = data[var].max().compute().item()

        table.add_row(
            var,
            f"{mean:.2f}",
            f"{std_dev:.2f}",
            f"{min_val:.2f}",
            f"{max_val:.2f}",
        )

    console.print(table)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyse dataset splits.")
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val"],
        default="train",
        help="Specify which dataset split to analyse (default: train)",
    )

    args = parser.parse_args()

    analyse_split(args.split)
