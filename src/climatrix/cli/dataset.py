from pathlib import Path

import cdsapi
import typer
from rich.console import Console
from rich.table import Table
from typing_extensions import Annotated

from climatrix.dataset.consts import DatasetType
from climatrix.io import load_request
from climatrix.models import Request

dataset_app = typer.Typer(name="Dataset commands")
console = Console()


def _get_default_years():
    return [2024]


def _get_default_months():
    return list(range(1, 13))


def _get_default_days():
    return list(range(1, 32))


def _get_default_hours():
    return list(range(0, 24))


@dataset_app.command("download", help="Download dataset")
def download_dataset(
    dataset: DatasetType,
    year: Annotated[
        list[int],
        typer.Option("--year", "-y", default_factory=_get_default_years),
    ],
    month: Annotated[
        list[int],
        typer.Option(
            "--month", "-m", min=1, max=12, default_factory=_get_default_months
        ),
    ],
    day: Annotated[
        list[int],
        typer.Option(
            "--day", "-d", min=1, max=31, default_factory=_get_default_days
        ),
    ],
    hour: Annotated[
        list[int],
        typer.Option(
            "--hour", "-h", min=0, max=23, default_factory=_get_default_hours
        ),
    ],
    target: Annotated[Path, typer.Option("--target", "-t")] = Path("."),
):
    request: Request = load_request(dataset)
    if target.is_dir():
        target /= request.filename
    if not target.parent.exists():
        target.parent.mkdir(parents=True)
    if target.exists():
        console.print(
            f"The file [bold green]{target}[/bold green] already exists"
        )
        return
    request.request["year"] = year
    request.request["month"] = month
    request.request["day"] = day
    request.request["time"] = hour
    with console.status("[magenta]Preparing request") as status:
        status.update("[magenta]Downloading dataset", spinner="bouncingBall")
        client = cdsapi.Client()
        client.retrieve(request.dataset, request.request).download(target)


@dataset_app.command("list", help="List available datasets")
def list_dataset():
    table = Table(title="List of datastets available in Climatrix")
    table.add_column("Dataset")

    for dataset in DatasetType:
        table.add_row(dataset.value)

    console.print(table)


# @dataset_app.command("show", help="Show the given dataset")
# def show(file: Path):
#     assert isinstance(file, Path)
#     file = file.expanduser().resolve()
#     if not file.exists():
#         raise FileNotFoundError(
#             f"The file [bold green]{file}[/bold green] does not exist"
#         )
#     with console.status("[magenta]Preparing dataset...") as status:
#         status.update("[magenta]Opening dataset...", spinner="bouncingBall")
#         dataset = BaseDataset.load(file)
#     dataset.plot()
