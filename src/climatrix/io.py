from uuid import uuid1
import sys
import os
import toml
from pathlib import Path
from importlib import resources
import importlib.util

import climatrix
from climatrix.models import Request
from climatrix.consts import Dataset

def get_resource_path(resource_path: str) -> Path:
    if (path:=  resources.files(climatrix.__name__).joinpath(resource_path)).exists():
        return path
    raise FileNotFoundError(f"The download script {resource_path} does not exists")

def get_download_request(dataset: Dataset):
    rel_path = Path(".").parents[1] / "scripts" / "download" / dataset
    return get_resource_path(rel_path)

def load_request(dataset: Dataset) -> Request:
    path: Path = get_resource_path(dataset.value)
    with open(path, "r") as f:
        return Request(**toml.load(f))