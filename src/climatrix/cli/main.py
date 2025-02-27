import typer 

from climatrix._version import __version__
from climatrix.cli.dataset import dataset_app

cm = typer.Typer(help="Climatrix CLI")
cm.add_typer(dataset_app, name="dataset")

@cm.command("version")
def version():
    print(__version__)


if  __name__ == "__main__":
    cm()