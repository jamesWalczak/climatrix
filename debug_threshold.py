#!/usr/bin/env python3
"""Debug the zero distance threshold issue."""
import sys
import os
sys.path.insert(0, '/home/runner/work/climatrix/climatrix/src')

import numpy as np
import xarray as xr
from climatrix.dataset.base import BaseClimatrixDataset
from climatrix.comparison import Comparison
from scipy.spatial import cKDTree


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


def debug_zero_threshold():
    """Debug the zero distance threshold issue."""
    print("Debugging zero distance threshold...")
    
    points1 = np.array([[0.0, 0.0], [1.0, 1.0]])
    values1 = np.array([1.0, 2.0])
    dataset1 = create_sparse_dataset(points1, values1, "predicted")
    
    # Same points
    points2 = np.array([[0.0, 0.0], [1.0, 1.0]])
    values2 = np.array([1.5, 2.5])
    dataset2 = create_sparse_dataset(points2, values2, "true")
    
    print(f"Points1: {points1}")
    print(f"Points2: {points2}")
    
    # Test with scipy directly
    tree = cKDTree(points2)
    distances, indices = tree.query(points1, distance_upper_bound=0.0)
    print(f"Distances: {distances}")
    print(f"Indices: {indices}")
    print(f"Valid mask (distances < inf): {distances < np.inf}")
    
    # The issue is likely that distance_upper_bound=0.0 is too strict
    # Let's try with a tiny tolerance
    distances_tiny, indices_tiny = tree.query(points1, distance_upper_bound=1e-10)
    print(f"Distances with tiny tolerance: {distances_tiny}")
    print(f"Valid mask with tiny tolerance: {distances_tiny < np.inf}")


if __name__ == "__main__":
    debug_zero_threshold()