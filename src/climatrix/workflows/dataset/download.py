from datetime import datetime, timedelta
import textwrap

import cdsapi
from prefect import flow, task

from climatrix.io import load_request
from climatrix.consts import Dataset
from climatrix.models import Request

@task(name="download-from-cds", tags=["dataset"])
def download_from_cds(dataset: Dataset, date: datetime):
    breakpoint()
    request: Request = load_request(dataset)
    client = cdsapi.Client()
    client.retrieve(dataset, request).download()    

@flow(name="prepare-dataset")
def prepare_dataset(dataset: Dataset, date: datetime) -> None:
    download_from_cds(dataset, date)

if __name__ == "__main__":
    prepare_dataset.deploy(
        "prepare_dataset",
        work_pool_name="default-pool",
    )