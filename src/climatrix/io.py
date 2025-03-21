import os
import tomllib as toml
from importlib import resources
from pathlib import Path

import climatrix
from climatrix.dataset.consts import DatasetType
from climatrix.models import Request


def get_resource_path(resource_path: str) -> Path:
    if (
        path := resources.files(climatrix.__name__).joinpath(resource_path)
    ).exists():
        return path
    raise FileNotFoundError(
        f"The download script {resource_path} does not exists"
    )


def get_download_request(dataset: DatasetType):
    rel_path = (
        Path(".") / ".." / ".." / "scripts" / "download" / f"{dataset}.toml"
    )
    return get_resource_path(rel_path)


def load_request(dataset: DatasetType) -> Request:
    path: Path = get_download_request(dataset)
    with open(path, "rb") as f:
        return Request(**toml.load(f))
