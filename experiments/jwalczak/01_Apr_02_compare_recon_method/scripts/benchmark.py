"""
This script benchmarks the time taken by different reconstruction methods
to process a set of input data. It uses the timeit module to measure
the execution time of each method and reports the results.

@author Jakub Walczak
"""

import argparse
import csv
import time
import tracemalloc
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

import climatrix as cm

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data"
RESULTS_DIR = SCRIPT_DIR.parent / "results"
TARGET_DIR = RESULTS_DIR / "benchmarking"
MMGN_CHECKPOINTS_DIR = SCRIPT_DIR.parent / "checkpoints" / "mmgn"
TARGET_DIR.mkdir(parents=True, exist_ok=True)
MMGN_CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)


def get_all_datasets_ids():
    return [
        p.stem.split("_")[-1]
        for p in DATA_DIR.glob("ecad_obs_europe_train_*.nc")
    ]


def load_hparams(method_name: str):
    if method_name == "mmgn":
        method_name = "inr/mmgn"
    return pd.read_csv(RESULTS_DIR / method_name / "hparams_summary.csv")


def load_data(dataset_id: str):
    return xr.open_dataset(
        DATA_DIR / f"ecad_obs_europe_train_{dataset_id}.nc"
    ).cm


def append_to_csv(file_path: Path, values: dict):
    file_exists = file_path.exists()
    with open(file_path, mode="a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=values.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(values)


def compute_target_domain(lat_min, lat_max, lon_min, lon_max, n: int):
    lats = np.random.uniform(lat_min, lat_max, n)
    lons = np.random.uniform(lon_min, lon_max, n)
    return xr.DataArray(
        data=np.ones(n, dtype=bool),
        coords={
            "latitude": (("points",), lats),
            "longitude": (("points",), lons),
        },
        dims=("points",),
    ).cm.domain


def benchmark_reconstruction_method(
    method_name: str,
    data,
    target_domain,
    hparams: dict,
    n: int,
    dataset_id: str,
):
    assert method_name in {
        "ok",
        "idw",
        "mmgn",
    }, f"Unknown method: {method_name}"

    start_time = time.perf_counter()
    data.reconstruct(target_domain, method=method_name, **hparams)
    end_time = time.perf_counter()

    tracemalloc.start()
    data.reconstruct(target_domain, method=method_name, **hparams)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return (end_time - start_time, peak)


def train_mmgn_if_needed(dataset_id: str, data, target_domain, hparams: dict):
    checkpoint_path = (
        MMGN_CHECKPOINTS_DIR / f"mmgn_checkpoint_{dataset_id}.pth"
    )
    if checkpoint_path.exists():
        print(
            f"MMGN checkpoint for dataset {dataset_id} already exists. Skipping training."
        )
        return

    print(f"Training MMGN for dataset {dataset_id}...")
    data.reconstruct(
        target_domain,
        method="mmgn",
        **hparams,
        checkpoint_path=checkpoint_path,
    )
    return checkpoint_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark reconstruction methods"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["ok", "idw", "mmgn"],
        default="all",
        help="Reconstruction method to benchmark",
    )
    return parser.parse_args()


def run():
    args = parse_args()
    method = args.method
    dataset_ids = get_all_datasets_ids()

    stats = {
        "time": [],
        "memory": [],
        "method": [],
        "n_points": [],
        "dataset_id": [],
    }
    for dataset_id in tqdm(dataset_ids):
        data = load_data(dataset_id)
        lat_min, lat_max = (
            data.domain.latitude.values.min().item(),
            data.domain.latitude.values.max().item(),
        )
        lon_min, lon_max = (
            data.domain.longitude.values.min().item(),
            data.domain.longitude.values.max().item(),
        )
        for n in [10, 100, 500, 1_000, 2_000, 5_000, 10_000]:
            target_domain = compute_target_domain(
                lat_min, lat_max, lon_min, lon_max, n=n
            )
            print(
                f"Benchmarking {method} with {n} target points on dataset {dataset_id}"
            )
            hparams_df = load_hparams(method).drop(
                columns=["opt_loss"], errors="ignore"
            )
            if method == "idw":
                hparams_df.drop(
                    columns=["Unnamed: 0"], inplace=True, errors="ignore"
                )
            hparams_df.set_index("dataset_id", inplace=True)
            hparams = hparams_df.loc[int(dataset_id)].to_dict()
            if method == "mmgn":
                hparams["num_epochs"] = 1  # reduce epochs for benchmarking
                checkpoint_path = train_mmgn_if_needed(
                    dataset_id, data, target_domain, hparams
                )
                hparams["checkpoint_path"] = checkpoint_path
            try:
                time, mem = benchmark_reconstruction_method(
                    method, data, target_domain, hparams, n, dataset_id
                )
            except Exception as e:
                print(
                    f"Error benchmarking {method} with {n} points on dataset {dataset_id}: {e}"
                )
                continue
            stats["time"].append(time)
            stats["memory"].append(mem)
            stats["method"].append(method)
            stats["n_points"].append(n)
            stats["dataset_id"].append(dataset_id)

    df = pd.DataFrame(stats)
    df.to_csv(TARGET_DIR / f"benchmark_results_{method}.csv", index=False)


if __name__ == "__main__":
    run()
