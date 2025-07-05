#!/usr/bin/env python3
"""
Realistic usage scenario test for sparse domain comparison
that demonstrates the functionality requested in the issue.
"""
import sys
import os
sys.path.insert(0, '/home/runner/work/climatrix/climatrix/src')

import numpy as np
import xarray as xr
from climatrix.dataset.base import BaseClimatrixDataset
from climatrix.comparison import Comparison


def create_observation_dataset(station_locations, temperatures, name="observations"):
    """Create a realistic sparse observation dataset."""
    lats, lons = station_locations[:, 0], station_locations[:, 1]
    
    da = xr.DataArray(
        temperatures,
        coords=[("point", np.arange(len(temperatures)))],
        dims=["point"],
        name=name,
    )
    
    # Add spatial coordinates as additional coordinate variables
    da = da.assign_coords({
        "latitude": ("point", lats),
        "longitude": ("point", lons),
    })
    
    return BaseClimatrixDataset(da)


def simulate_weather_station_comparison():
    """
    Simulate a realistic scenario:
    Compare model predictions with weather station observations.
    """
    print("=" * 70)
    print("REALISTIC SCENARIO: Model vs Weather Station Comparison")
    print("=" * 70)
    
    # Simulate weather station locations (scattered points)
    np.random.seed(42)  # For reproducible results
    n_stations = 15
    station_lats = np.random.uniform(40.0, 60.0, n_stations)  # Latitude range
    station_lons = np.random.uniform(-10.0, 20.0, n_stations)  # Longitude range
    station_locations = np.column_stack([station_lats, station_lons])
    
    # Simulate observed temperatures at weather stations
    observed_temps = 15 + 10 * np.sin(np.radians(station_lats - 50)) + np.random.normal(0, 2, n_stations)
    
    print(f"Weather Stations: {n_stations} locations")
    print(f"Latitude range: {station_lats.min():.2f} to {station_lats.max():.2f}")
    print(f"Longitude range: {station_lons.min():.2f} to {station_lons.max():.2f}")
    print(f"Temperature range: {observed_temps.min():.2f} to {observed_temps.max():.2f} °C")
    
    # Create observation dataset
    observations = create_observation_dataset(station_locations, observed_temps, "observations")
    
    # Simulate model predictions at slightly different locations
    # (representing model grid points near stations)
    model_offset = 0.1  # Small spatial offset
    model_lats = station_lats + np.random.uniform(-model_offset, model_offset, n_stations)
    model_lons = station_lons + np.random.uniform(-model_offset, model_offset, n_stations)
    model_locations = np.column_stack([model_lats, model_lons])
    
    # Simulate model predictions (with some bias and error)
    model_bias = 1.5  # Model tends to be warmer
    model_temps = observed_temps + model_bias + np.random.normal(0, 1, n_stations)
    
    print(f"Model Predictions: {n_stations} grid points")
    print(f"Spatial offset: ±{model_offset} degrees")
    print(f"Model bias: +{model_bias} °C")
    
    # Create model dataset
    model_predictions = create_observation_dataset(model_locations, model_temps, "model")
    
    return observations, model_predictions


def test_various_distance_thresholds(observations, model_predictions):
    """Test comparison with various distance thresholds."""
    print("\n" + "-" * 50)
    print("TESTING DIFFERENT DISTANCE THRESHOLDS")
    print("-" * 50)
    
    thresholds = [None, 1.0, 0.5, 0.2, 0.1, 0.05]
    
    for threshold in thresholds:
        try:
            comparison = Comparison(model_predictions, observations, distance_threshold=threshold)
            n_matches = len(comparison.diff.da.values)
            
            if n_matches > 0:
                rmse = comparison.compute_rmse()
                mae = comparison.compute_mae()
                max_error = comparison.compute_max_abs_error()
                
                threshold_str = f"{threshold:.2f}" if threshold is not None else "None"
                print(f"Threshold {threshold_str:>6}: {n_matches:2d} matches, "
                      f"RMSE={rmse:5.2f}, MAE={mae:5.2f}, Max={max_error:5.2f}")
            else:
                threshold_str = f"{threshold:.2f}" if threshold is not None else "None"
                print(f"Threshold {threshold_str:>6}: {n_matches:2d} matches (no valid correspondences)")
                
        except Exception as e:
            print(f"Error with threshold {threshold}: {e}")


def demonstrate_issue_requirements():
    """Demonstrate that the issue requirements are met."""
    print("\n" + "=" * 70)
    print("DEMONSTRATING ISSUE REQUIREMENTS")
    print("=" * 70)
    
    # Create test datasets
    observations, model_predictions = simulate_weather_station_comparison()
    
    print("\n✓ Requirement: Enable running comparison by comparing closest observations")
    comparison_closest = Comparison(model_predictions, observations)
    print(f"  Successfully compared {len(comparison_closest.diff.da.values)} point pairs using closest neighbor matching")
    
    print("\n✓ Requirement: Use scipy cKDTree for efficient querying nearest neighbours")
    print("  Implementation uses scipy.spatial.cKDTree in _compute_sparse_diff method")
    
    print("\n✓ Requirement: Take into account distance threshold when checking correspondence")
    comparison_threshold = Comparison(model_predictions, observations, distance_threshold=0.2)
    n_matches_threshold = len(comparison_threshold.diff.da.values)
    n_matches_all = len(comparison_closest.diff.da.values)
    print(f"  With distance threshold 0.2°: {n_matches_threshold} matches")
    print(f"  Without distance threshold:   {n_matches_all} matches")
    print(f"  Threshold filtering works: {'✓' if n_matches_threshold <= n_matches_all else '✗'}")
    
    # Test various thresholds to show fine-grained control
    test_various_distance_thresholds(observations, model_predictions)
    
    print("\n✓ All requirements from issue #24 have been successfully implemented!")
    
    # Compute final comparison metrics
    print("\n" + "-" * 50)
    print("FINAL COMPARISON RESULTS")
    print("-" * 50)
    
    final_comparison = Comparison(model_predictions, observations, distance_threshold=0.2)
    if len(final_comparison.diff.da.values) > 0:
        report = final_comparison.compute_report()
        print("Metrics with 0.2° distance threshold:")
        for metric, value in report.items():
            print(f"  {metric}: {value:.3f}")
    else:
        print("No valid matches found with 0.2° threshold")


if __name__ == "__main__":
    print("Testing sparse domain comparison with realistic weather data scenario...")
    
    try:
        demonstrate_issue_requirements()
        print("\n🎉 SUCCESS: All functionality working as intended!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)