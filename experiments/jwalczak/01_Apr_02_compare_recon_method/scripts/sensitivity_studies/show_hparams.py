from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()


def read_idw_hparams():
    hparam_file = (
        Path(__file__).parent.parent.parent
        / "results"
        / "idw"
        / "hparams_summary.csv"
    )
    return pd.read_csv(hparam_file)


def read_ok_hparams():
    hparam_file = (
        Path(__file__).parent.parent.parent
        / "results"
        / "ok"
        / "hparams_summary.csv"
    )
    return pd.read_csv(hparam_file)


def read_mmgn_hparams():
    hparam_file = (
        Path(__file__).parent.parent.parent
        / "results"
        / "inr"
        / "mmgn"
        / "hparams_summary.csv"
    )
    return pd.read_csv(hparam_file)


def summarize_params(df: pd.DataFrame, method_name: str):
    console.print(f"[bold]{method_name} Hyperparameters Summary[/bold]")
    table = Table(title=f"{method_name} Hyperparameters Summary")
    table.add_column("Hyperparameter", justify="left")
    table.add_column("Mean", justify="right")
    table.add_column("Mode", justify="right")
    table.add_column("Median", justify="right")

    for column in df.columns:
        if column == "dataset_id":
            continue
        mode = df[column].mode().iloc[0]
        mode = f"{mode}"
        try:
            mean = df[column].mean()
            mean = f"{mean:.2f}"
        except TypeError:
            mean = "-"
        try:
            median = df[column].median()
            median = f"{median:.2f}"
        except TypeError:
            median = "-"
        table.add_row(column, mean, mode, median)
    console.print(table)


if __name__ == "__main__":
    idw_hparams = read_idw_hparams()
    ok_hparams = read_ok_hparams()
    mmgn_hparams = read_mmgn_hparams()

    summarize_params(idw_hparams, "IDW")
    summarize_params(ok_hparams, "OK")
    summarize_params(mmgn_hparams, "MMGN")
