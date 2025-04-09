import glob

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

ROOT_EXP_DIR = "results/ordinary_kriging"
UNIFORM_EXP_DIR = f"{ROOT_EXP_DIR}/uniform"
NORMAL_EXP_DIR = f"{ROOT_EXP_DIR}/normal"

normal_metric_files = glob.glob(f"{NORMAL_EXP_DIR}/*/metrics.csv")


def analyse_uniform():
    uniform_metric_files = glob.glob(f"{UNIFORM_EXP_DIR}/*/metrics.csv")
    if len(uniform_metric_files) == 0:
        return
    uniform_metrics = pd.concat(
        [pd.read_csv(f) for f in uniform_metric_files], ignore_index=True
    )
    table = Table(
        title="Metrics for reconstruction from uniformly sampled "
        "data (Ordinary Kriging)"
    )
    table.add_column("Metric")
    table.add_column("Mean value")
    table.add_column("Std value")
    table.add_column("Std error")

    for col in uniform_metrics.columns:
        std_err = uniform_metrics[col].std() / np.sqrt(len(uniform_metrics))
        table.add_row(
            col,
            f"{uniform_metrics[col].mean():.4f}",
            f"{uniform_metrics[col].std():.4f}",
            f"{std_err:.4f}",
        )
    console = Console()
    console.print(table)


def analyse_normal():
    normal_metric_files = glob.glob(f"{NORMAL_EXP_DIR}/*/metrics.csv")
    if len(normal_metric_files) == 0:
        return
    normal_metrics = pd.concat(
        [pd.read_csv(f) for f in normal_metric_files], ignore_index=True
    )
    table = Table(
        title="Metrics for reconstruction from uniformly sampled "
        "data (Ordinary Kriging)"
    )
    for col in normal_metrics.columns:
        table.add_column(f"mean {col}")
        table.add_column(f"std {col}")

    for col in normal_metrics.columns:
        table.add_row(
            f"{normal_metrics[col].mean():.4f}",
            f"{normal_metrics[col].std():.4f}",
        )
    console = Console()
    console.print(table)


if __name__ == "__main__":
    analyse_uniform()
    analyse_normal()
