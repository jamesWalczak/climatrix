import tomllib as toml
from importlib import resources
from pathlib import Path

import climatrix
from climatrix.dataset.consts import DatasetType
from climatrix.models import Request


def get_resource_path(resource_path: str) -> Path:
    """
    Return the path to the resource file.

    Parameters
    ----------
    resource_path : str
        The path to the resource file, relative to the package root.

    Returns
    -------
    path : Path
        The absolute path to the resource file.

    Raises
    ------
    FileNotFoundError
        If the resource file does not exist.
    """
    if (
        path := resources.files(climatrix.__name__).joinpath(resource_path)
    ).exists():
        return path
    raise FileNotFoundError(
        f"The download script {resource_path} does not exists"
    )


def get_download_request(dataset: DatasetType) -> Path:
    """
    Return the path to the TOML file with the download request.

    Parameters
    ----------
    dataset : DatasetType
        The type of the dataset to download.

    Returns
    -------
    path : Path
        The absolute path to the TOML file with the download request.
    """
    rel_path = (
        Path("resources") / "scripts" / "download" / "cds" / f"{dataset}.toml"
    )
    return get_resource_path(rel_path)


def load_request(dataset: DatasetType) -> Request:
    """
    Load a TOML file containing a download request.

    Parameters
    ----------
    dataset : DatasetType
        The type of the dataset to download.

    Returns
    -------
    request : Request
        The loaded request.
    """
    path: Path = get_download_request(dataset)
    with open(path, "rb") as f:
        return Request(**toml.load(f))
