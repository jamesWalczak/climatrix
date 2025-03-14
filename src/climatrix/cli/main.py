import typer

from climatrix._version import __version__
from climatrix.cli.dataset import dataset_app

cm = typer.Typer(help="Climatrix CLI")
cm.add_typer(dataset_app, name="dataset")


def version_callback(value: bool):
    if value:
        print(f"Climatrix version: {__version__}")
        raise typer.Exit()


@cm.command("version")
def version():
    version_callback(True)


@cm.callback()
def main(
    version: bool = typer.Option(
        None, "--version", callback=version_callback, is_eager=True
    ),
):
    return


if __name__ == "__main__":
    cm()
