#!/usr/bin/env python3
"""
Test script to reproduce the current limitation with sparse domain comparison
and test the new functionality.
"""
import sys
import os
sys.path.insert(0, '/home/runner/work/climatrix/climatrix/src')

import numpy as np
import xarray as xr
from climatrix.dataset.base import BaseClimatrixDataset
from climatrix.comparison import Comparison


def create_sparse_dataset(points, values, name="test"):
    """Create a sparse dataset from points and values."""
    lats, lons = points[:, 0], points[:, 1]
    
    da = xr.DataArray(
        values,
        coords=[("point", np.arange(len(values)))],
        dims=["point"],
        name=name,
    )
    
    # Add spatial coordinates as additional coordinate variables
    da = da.assign_coords({
        "latitude": ("point", lats),
        "longitude": ("point", lons),
    })
    
    return BaseClimatrixDataset(da)


def test_current_limitation():
    """Test that sparse domain comparison currently fails."""
    print("Testing current limitation with sparse domains...")
    
    # Create two sparse datasets
    points1 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    values1 = np.array([1.0, 2.0, 3.0])
    dataset1 = create_sparse_dataset(points1, values1, "predicted")
    
    points2 = np.array([[0.1, 0.1], [1.1, 1.1], [2.1, 2.1]])
    values2 = np.array([1.5, 2.5, 3.5])
    dataset2 = create_sparse_dataset(points2, values2, "true")
    
    print(f"Dataset 1 is_sparse: {dataset1.domain.is_sparse}")
    print(f"Dataset 2 is_sparse: {dataset2.domain.is_sparse}")
    
    try:
        comparison = Comparison(dataset1, dataset2)
        print("SUCCESS: Sparse domain comparison worked!")
        print(f"Difference values: {comparison.diff.da.values}")
    except Exception as e:
        print(f"EXPECTED FAILURE: {e}")
        return False
    
    return True


def test_dense_domain_still_works():
    """Test that dense domain comparison still works."""
    print("\nTesting that dense domain comparison still works...")
    
    lat = np.array([-45.0, 0.0, 45.0])
    lon = np.array([0.0, 180.0, 360.0])
    
    predicted_data = np.arange(9).reshape(3, 3) + 1.0
    predicted_da = xr.DataArray(
        predicted_data,
        coords=[("lat", lat), ("lon", lon)],
        name="predicted",
    )
    predicted_dataset = BaseClimatrixDataset(predicted_da)
    
    true_data = np.arange(9).reshape(3, 3) + 0.5
    true_da = xr.DataArray(
        true_data,
        coords=[("lat", lat), ("lon", lon)],
        name="true",
    )
    true_dataset = BaseClimatrixDataset(true_da)
    
    try:
        comparison = Comparison(predicted_dataset, true_dataset)
        print("SUCCESS: Dense domain comparison still works!")
        print(f"RMSE: {comparison.compute_rmse():.3f}")
        return True
    except Exception as e:
        print(f"UNEXPECTED FAILURE: {e}")
        return False


if __name__ == "__main__":
    print("Testing current state of comparison module...")
    
    dense_works = test_dense_domain_still_works()
    sparse_works = test_current_limitation()
    
    print(f"\nResults:")
    print(f"Dense domain comparison works: {dense_works}")
    print(f"Sparse domain comparison works: {sparse_works}")