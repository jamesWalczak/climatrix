import logging
import os
import re

import requests
from tqdm import tqdm

log = logging.getLogger(__name__)


def get_zenodo_download_url(doi: str, filename: str):
    """Get the direct download URL for a file from Zenodo using DOI"""

    # Extract record ID from DOI
    # DOI format: 10.5281/zenodo.1234567
    match = re.search(r"zenodo\.(\d+)", doi)
    if not match:
        log.error(f"Invalid Zenodo DOI format: {doi}")
        return None

    record_id = match.group(1)

    # Get record metadata from Zenodo API
    api_url = f"https://zenodo.org/api/records/{record_id}"
    log.info(f"Fetching metadata from: {api_url}")

    try:
        response = requests.get(api_url)
        if response.status_code != 200:
            log.error(
                f"Failed to fetch Zenodo metadata. Status: {response.status_code}"
            )
            return None

        data = response.json()

        for file_info in data.get("files", []):
            if file_info["key"] == filename:
                return file_info["links"]["self"], file_info["size"]

        log.error(f"File '{filename}' not found in Zenodo record {record_id}")
        return None

    except requests.RequestException as e:
        log.error(f"Network error while fetching Zenodo metadata: {e}")
        return None
    except (KeyError, ValueError) as e:
        log.error(f"Error parsing Zenodo metadata: {e}")
        return None


def download_zenodo_file(doi, filename, local_path):
    """Download a file from Zenodo using DOI with progress bar"""

    log = logging.getLogger(__name__)

    try:
        url_info = get_zenodo_download_url(doi, filename)
        if url_info is None:
            return False

        download_url, file_size = url_info
        log.info(f"Downloading from: {download_url}")

        with requests.get(download_url, stream=True) as response:
            if response.status_code != 200:
                log.error(
                    f"Failed to download file. Status: {response.status_code}"
                )
                return False

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            # Set up progress bar
            total_size = file_size or int(
                response.headers.get("content-length", 0)
            )

            with open(local_path, "wb") as f:
                with tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"Downloading {filename}",
                ) as pbar:

                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:  # Filter out keep-alive chunks
                            f.write(chunk)
                            pbar.update(len(chunk))

            log.info(f"Successfully downloaded {filename} to {local_path}")
            return True

    except requests.RequestException as e:
        log.error(f"Network error while downloading {filename}: {e}")
        return False
    except OSError as e:
        log.error(f"File I/O error while downloading {filename}: {e}")
        return False
    except Exception as e:
        log.error(f"Unexpected error while downloading {filename}: {e}")
        return False
