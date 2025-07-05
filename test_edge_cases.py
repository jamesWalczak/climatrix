#!/usr/bin/env python3
"""
Test edge cases for sparse domain comparison functionality.
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


def test_single_point_datasets():
    """Test comparison with single point datasets."""
    print("Testing single point datasets...")
    
    points1 = np.array([[0.0, 0.0]])
    values1 = np.array([5.0])
    dataset1 = create_sparse_dataset(points1, values1, "predicted")
    
    points2 = np.array([[0.0, 0.0]])
    values2 = np.array([3.0])
    dataset2 = create_sparse_dataset(points2, values2, "true")
    
    comparison = Comparison(dataset1, dataset2)
    
    expected_diff = 2.0
    assert abs(comparison.diff.da.values[0] - expected_diff) < 1e-10
    assert abs(comparison.compute_rmse() - 2.0) < 1e-10
    assert abs(comparison.compute_mae() - 2.0) < 1e-10
    
    print("✓ Single point comparison works correctly")
    return True


def test_identical_datasets():
    """Test comparison of identical sparse datasets."""
    print("\nTesting identical datasets...")
    
    points = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    values = np.array([1.0, 2.0, 3.0])
    
    dataset1 = create_sparse_dataset(points, values, "predicted")
    dataset2 = create_sparse_dataset(points, values, "true")
    
    comparison = Comparison(dataset1, dataset2)
    
    expected_diff = np.array([0.0, 0.0, 0.0])
    np.testing.assert_array_almost_equal(comparison.diff.da.values, expected_diff)
    
    assert abs(comparison.compute_rmse()) < 1e-10
    assert abs(comparison.compute_mae()) < 1e-10
    assert abs(comparison.compute_max_abs_error()) < 1e-10
    
    print("✓ Identical datasets produce zero differences")
    return True


def test_nan_values():
    """Test handling of NaN values in sparse datasets."""
    print("\nTesting NaN value handling...")
    
    points = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    values1 = np.array([1.0, np.nan, 3.0])
    values2 = np.array([1.5, 2.5, np.nan])
    
    dataset1 = create_sparse_dataset(points, values1, "predicted")
    dataset2 = create_sparse_dataset(points, values2, "true")
    
    comparison = Comparison(dataset1, dataset2)
    
    # Should compute differences: [1.0-1.5, nan-2.5, 3.0-nan] = [-0.5, nan, nan]
    diffs = comparison.diff.da.values
    assert abs(diffs[0] - (-0.5)) < 1e-10
    assert np.isnan(diffs[1])
    assert np.isnan(diffs[2])
    
    # Metrics should handle NaN values appropriately
    rmse = comparison.compute_rmse()
    mae = comparison.compute_mae()
    
    # Should compute metrics based on non-NaN values only
    assert abs(rmse - 0.5) < 1e-10  # Only one valid difference: -0.5
    assert abs(mae - 0.5) < 1e-10
    
    print("✓ NaN values handled correctly in sparse comparison")
    return True


def test_very_large_distance_threshold():
    """Test with very large distance threshold."""
    print("\nTesting very large distance threshold...")
    
    # Create datasets with points very far apart
    points1 = np.array([[0.0, 0.0], [1.0, 1.0]])
    values1 = np.array([1.0, 2.0])
    dataset1 = create_sparse_dataset(points1, values1, "predicted")
    
    points2 = np.array([[100.0, 100.0], [101.0, 101.0]])
    values2 = np.array([1.5, 2.5])
    dataset2 = create_sparse_dataset(points2, values2, "true")
    
    # Very large threshold should still find matches
    comparison = Comparison(dataset1, dataset2, distance_threshold=1000.0)
    
    assert len(comparison.diff.da.values) == 2
    print("✓ Very large distance threshold works correctly")
    return True


def test_zero_distance_threshold():
    """Test with zero distance threshold."""
    print("\nTesting zero distance threshold...")
    
    points1 = np.array([[0.0, 0.0], [1.0, 1.0]])
    values1 = np.array([1.0, 2.0])
    dataset1 = create_sparse_dataset(points1, values1, "predicted")
    
    # Same points
    points2 = np.array([[0.0, 0.0], [1.0, 1.0]])
    values2 = np.array([1.5, 2.5])
    dataset2 = create_sparse_dataset(points2, values2, "true")
    
    # Zero threshold should still match exact points
    comparison = Comparison(dataset1, dataset2, distance_threshold=0.0)
    
    assert len(comparison.diff.da.values) == 2
    print("✓ Zero distance threshold works for exact matches")
    return True


def test_plotting_functionality():
    """Test that plotting still works with sparse domains."""
    print("\nTesting plotting functionality...")
    
    points = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    values1 = np.array([1.0, 2.0, 3.0])
    values2 = np.array([1.5, 2.5, 3.5])
    
    dataset1 = create_sparse_dataset(points, values1, "predicted")
    dataset2 = create_sparse_dataset(points, values2, "true")
    
    comparison = Comparison(dataset1, dataset2)
    
    # Should be able to call plotting methods without error
    try:
        # Test plot_diff (just check it doesn't crash, don't actually display)
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        ax = comparison.plot_diff(show=False)
        assert ax is not None
        
        # Test signed diff histogram
        ax_hist = comparison.plot_signed_diff_hist()
        assert ax_hist is not None
        
        print("✓ Plotting functionality works with sparse domains")
        return True
        
    except Exception as e:
        print(f"✗ Plotting failed: {e}")
        return False


def run_edge_case_tests():
    """Run all edge case tests."""
    tests = [
        test_single_point_datasets,
        test_identical_datasets,
        test_nan_values,
        test_very_large_distance_threshold,
        test_zero_distance_threshold,
        test_plotting_functionality,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print(f"\n{'='*60}")
    print(f"Edge Case Test Results: {sum(results)}/{len(results)} passed")
    
    if all(results):
        print("🎉 All edge case tests passed!")
        return True
    else:
        print("❌ Some edge case tests failed")
        return False


if __name__ == "__main__":
    print("Running edge case tests for sparse domain comparison...")
    print("="*60)
    
    success = run_edge_case_tests()
    sys.exit(0 if success else 1)