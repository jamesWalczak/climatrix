#!/usr/bin/env python3
"""
Comprehensive tests for sparse domain comparison functionality.
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


def test_sparse_comparison_exact_match():
    """Test sparse domain comparison with exactly matching points."""
    print("Testing sparse comparison with exact spatial matches...")
    
    points = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    values1 = np.array([1.0, 2.0, 3.0])
    values2 = np.array([1.5, 2.5, 3.5])
    
    dataset1 = create_sparse_dataset(points, values1, "predicted")
    dataset2 = create_sparse_dataset(points, values2, "true")
    
    comparison = Comparison(dataset1, dataset2)
    expected_diff = values1 - values2  # [-0.5, -0.5, -0.5]
    
    assert np.allclose(comparison.diff.da.values, expected_diff), \
        f"Expected {expected_diff}, got {comparison.diff.da.values}"
    
    print(f"✓ Differences: {comparison.diff.da.values}")
    print(f"✓ RMSE: {comparison.compute_rmse():.3f}")
    return True


def test_sparse_comparison_with_distance_threshold():
    """Test sparse domain comparison with distance threshold."""
    print("\nTesting sparse comparison with distance threshold...")
    
    # Dataset 1: points at [0,0], [1,1], [2,2]
    points1 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    values1 = np.array([1.0, 2.0, 3.0])
    dataset1 = create_sparse_dataset(points1, values1, "predicted")
    
    # Dataset 2: points slightly offset at [0.1,0.1], [1.1,1.1], [2.1,2.1]
    points2 = np.array([[0.1, 0.1], [1.1, 1.1], [2.1, 2.1]])
    values2 = np.array([1.5, 2.5, 3.5])
    dataset2 = create_sparse_dataset(points2, values2, "true")
    
    # Test with large threshold (should match all points)
    comparison_large = Comparison(dataset1, dataset2, distance_threshold=1.0)
    assert len(comparison_large.diff.da.values) == 3, \
        f"Expected 3 matches with large threshold, got {len(comparison_large.diff.da.values)}"
    print(f"✓ Large threshold (1.0): {len(comparison_large.diff.da.values)} matches")
    
    # Test with small threshold (should match no points)
    comparison_small = Comparison(dataset1, dataset2, distance_threshold=0.1)
    assert len(comparison_small.diff.da.values) == 0, \
        f"Expected 0 matches with small threshold, got {len(comparison_small.diff.da.values)}"
    print(f"✓ Small threshold (0.1): {len(comparison_small.diff.da.values)} matches")
    
    # Test with medium threshold (should match all points since distance is ~0.14)
    comparison_medium = Comparison(dataset1, dataset2, distance_threshold=0.2)
    assert len(comparison_medium.diff.da.values) == 3, \
        f"Expected 3 matches with medium threshold, got {len(comparison_medium.diff.da.values)}"
    print(f"✓ Medium threshold (0.2): {len(comparison_medium.diff.da.values)} matches")
    
    return True


def test_sparse_comparison_different_sizes():
    """Test sparse domain comparison with different dataset sizes."""
    print("\nTesting sparse comparison with different dataset sizes...")
    
    # Dataset 1: 5 points
    points1 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    values1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    dataset1 = create_sparse_dataset(points1, values1, "predicted")
    
    # Dataset 2: 3 points, subset of dataset1 locations
    points2 = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    values2 = np.array([2.5, 3.5, 4.5])
    dataset2 = create_sparse_dataset(points2, values2, "true")
    
    comparison = Comparison(dataset1, dataset2)
    
    # Should find 5 matches (all points in dataset1 get matched to nearest in dataset2)
    assert len(comparison.diff.da.values) == 5, \
        f"Expected 5 matches, got {len(comparison.diff.da.values)}"
    
    print(f"✓ Different sizes: {len(comparison.diff.da.values)} matches from 5 vs 3 points")
    print(f"✓ Differences: {comparison.diff.da.values}")
    
    return True


def test_metrics_computation():
    """Test that metric computations work correctly for sparse domains."""
    print("\nTesting metric computations for sparse domains...")
    
    points = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    values1 = np.array([1.0, 2.0, 3.0])
    values2 = np.array([1.5, 2.5, 3.5])
    
    dataset1 = create_sparse_dataset(points, values1, "predicted")
    dataset2 = create_sparse_dataset(points, values2, "true")
    
    comparison = Comparison(dataset1, dataset2)
    
    # Test individual metrics
    rmse = comparison.compute_rmse()
    mae = comparison.compute_mae()
    max_abs_error = comparison.compute_max_abs_error()
    r2 = comparison.compute_r2()
    
    # Expected values for differences [-0.5, -0.5, -0.5]
    expected_rmse = 0.5
    expected_mae = 0.5
    expected_max_abs_error = 0.5
    
    assert abs(rmse - expected_rmse) < 1e-10, f"RMSE: expected {expected_rmse}, got {rmse}"
    assert abs(mae - expected_mae) < 1e-10, f"MAE: expected {expected_mae}, got {mae}"
    assert abs(max_abs_error - expected_max_abs_error) < 1e-10, \
        f"Max abs error: expected {expected_max_abs_error}, got {max_abs_error}"
    
    print(f"✓ RMSE: {rmse:.3f}")
    print(f"✓ MAE: {mae:.3f}")
    print(f"✓ Max Abs Error: {max_abs_error:.3f}")
    print(f"✓ R²: {r2:.3f}")
    
    # Test report
    report = comparison.compute_report()
    assert len(report) == 4, f"Expected 4 metrics in report, got {len(report)}"
    print(f"✓ Report contains all metrics: {list(report.keys())}")
    
    return True


def test_mixed_domain_error():
    """Test that comparison between sparse and dense domains raises appropriate error."""
    print("\nTesting error handling for mixed domain types...")
    
    # Create sparse dataset
    points = np.array([[0.0, 0.0], [1.0, 1.0]])
    values = np.array([1.0, 2.0])
    sparse_dataset = create_sparse_dataset(points, values, "sparse")
    
    # Create dense dataset
    lat = np.array([-45.0, 0.0, 45.0])
    lon = np.array([0.0, 180.0, 360.0])
    dense_data = np.arange(9).reshape(3, 3) + 1.0
    dense_da = xr.DataArray(
        dense_data,
        coords=[("lat", lat), ("lon", lon)],
        name="dense",
    )
    dense_dataset = BaseClimatrixDataset(dense_da)
    
    try:
        comparison = Comparison(sparse_dataset, dense_dataset)
        print("✗ Expected error for mixed domain types")
        return False
    except ValueError as e:
        expected_msg = "Comparison between sparse and dense domains is not supported"
        if expected_msg in str(e):
            print(f"✓ Correctly raised error: {e}")
            return True
        else:
            print(f"✗ Unexpected error message: {e}")
            return False


def test_empty_result_handling():
    """Test handling when no points match within distance threshold."""
    print("\nTesting empty result handling...")
    
    # Create datasets with points far apart
    points1 = np.array([[0.0, 0.0], [1.0, 1.0]])
    values1 = np.array([1.0, 2.0])
    dataset1 = create_sparse_dataset(points1, values1, "predicted")
    
    points2 = np.array([[10.0, 10.0], [11.0, 11.0]])  # Far away
    values2 = np.array([1.5, 2.5])
    dataset2 = create_sparse_dataset(points2, values2, "true")
    
    # Use very small distance threshold
    comparison = Comparison(dataset1, dataset2, distance_threshold=0.1)
    
    assert len(comparison.diff.da.values) == 0, \
        f"Expected 0 matches with small threshold and distant points, got {len(comparison.diff.da.values)}"
    
    print(f"✓ Empty result handled correctly: {len(comparison.diff.da.values)} matches")
    return True


def run_all_tests():
    """Run all tests and report results."""
    tests = [
        test_sparse_comparison_exact_match,
        test_sparse_comparison_with_distance_threshold,
        test_sparse_comparison_different_sizes,
        test_metrics_computation,
        test_mixed_domain_error,
        test_empty_result_handling,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} failed: {e}")
            results.append(False)
    
    print(f"\n{'='*60}")
    print(f"Test Results: {sum(results)}/{len(results)} passed")
    
    if all(results):
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed")
        return False


if __name__ == "__main__":
    print("Running comprehensive sparse domain comparison tests...")
    print("="*60)
    
    success = run_all_tests()
    sys.exit(0 if success else 1)