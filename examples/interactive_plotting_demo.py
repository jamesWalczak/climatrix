#!/usr/bin/env python3
"""
Example script demonstrating the interactive plotting functionality.

This script creates dummy climate datasets and shows how to use the
interactive plotting utility with different data types and dimensions.
"""

import numpy as np
import xarray as xr

# Try to import climatrix
try:
    import climatrix as cm
    print("✓ Climatrix imported successfully")
except ImportError:
    print("✗ Could not import climatrix. Make sure it's installed.")
    exit(1)


def create_dummy_dense_dataset():
    """Create a dummy dense climate dataset with time and spatial dimensions."""
    print("\nCreating dummy dense dataset...")
    
    # Create coordinate arrays
    lats = np.linspace(-90, 90, 50)
    lons = np.linspace(-180, 180, 100)
    time = np.arange(np.datetime64('2020-01-01'), np.datetime64('2020-12-31'), np.timedelta64(1, 'D'))
    
    # Create meshgrid for spatial coordinates
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
    
    # Generate dummy temperature data with temporal and spatial variation
    np.random.seed(42)
    
    # Base temperature pattern (warmer near equator)
    base_temp = 15 - 0.7 * np.abs(lat_grid)
    
    # Add seasonal variation
    temp_data = np.zeros((len(time), len(lats), len(lons)))
    for i, t in enumerate(time):
        day_of_year = (t - np.datetime64('2020-01-01')) / np.timedelta64(1, 'D')
        seasonal_factor = 10 * np.cos(2 * np.pi * day_of_year / 365.25)
        
        # Add latitude-dependent seasonal effect
        seasonal_temp = base_temp + seasonal_factor * np.cos(np.radians(lat_grid))
        
        # Add some random noise
        noise = np.random.normal(0, 2, seasonal_temp.shape)
        temp_data[i] = seasonal_temp + noise
    
    # Create xarray dataset
    ds = xr.Dataset(
        {
            'temperature': (['time', 'latitude', 'longitude'], temp_data)
        },
        coords={
            'time': time,
            'latitude': lats,
            'longitude': lons
        },
        attrs={
            'title': 'Dummy Global Temperature Dataset',
            'description': 'Synthetic daily temperature data for demonstration'
        }
    )
    
    ds.temperature.attrs = {
        'units': 'degrees_C',
        'long_name': 'Air Temperature',
        'standard_name': 'air_temperature'
    }
    
    print(f"✓ Dense dataset created: {ds.sizes}")
    return ds


def create_dummy_sparse_dataset():
    """Create a dummy sparse climate dataset with station data."""
    print("\nCreating dummy sparse dataset...")
    
    # Create station locations (sparse points)
    np.random.seed(123)
    n_stations = 50
    
    station_lats = np.random.uniform(-90, 90, n_stations)
    station_lons = np.random.uniform(-180, 180, n_stations)
    station_ids = [f"STATION_{i:03d}" for i in range(n_stations)]
    
    # Create time series
    time = np.arange(np.datetime64('2020-01-01'), np.datetime64('2020-01-31'), np.timedelta64(1, 'D'))
    
    # Generate dummy precipitation data
    precip_data = np.zeros((len(time), n_stations))
    for i, (lat, lon) in enumerate(zip(station_lats, station_lons)):
        # Base precipitation depends on latitude (more in tropics)
        base_precip = 5 + 10 * np.exp(-0.02 * lat**2)
        
        # Add random variability
        for j in range(len(time)):
            if np.random.random() < 0.3:  # Rain on 30% of days
                precip_data[j, i] = np.random.exponential(base_precip)
            else:
                precip_data[j, i] = 0
    
    # Create xarray dataset
    ds = xr.Dataset(
        {
            'precipitation': (['time', 'station'], precip_data)
        },
        coords={
            'time': time,
            'station': station_ids,
            'latitude': ('station', station_lats),
            'longitude': ('station', station_lons)
        },
        attrs={
            'title': 'Dummy Station Precipitation Dataset',
            'description': 'Synthetic station precipitation data for demonstration'
        }
    )
    
    ds.precipitation.attrs = {
        'units': 'mm/day',
        'long_name': 'Daily Precipitation',
        'standard_name': 'precipitation_flux'
    }
    
    print(f"✓ Sparse dataset created: {ds.sizes}")
    return ds


def create_dummy_3d_dataset():
    """Create a dummy 3D climate dataset with vertical levels."""
    print("\nCreating dummy 3D dataset...")
    
    # Create coordinates
    lats = np.linspace(-60, 60, 30)
    lons = np.linspace(-120, 120, 60)
    levels = np.array([1000, 850, 700, 500, 300, 200, 100])  # Pressure levels in hPa
    time = np.arange(np.datetime64('2020-01-01'), np.datetime64('2020-01-08'), np.timedelta64(1, 'D'))
    
    # Generate dummy wind speed data
    np.random.seed(456)
    wind_data = np.zeros((len(time), len(levels), len(lats), len(lons)))
    
    for t in range(len(time)):
        for k, level in enumerate(levels):
            # Wind speed increases with height and varies with latitude
            base_wind = 10 + (1000 - level) / 50  # Stronger at higher altitudes
            
            for i, lat in enumerate(lats):
                # Jet stream effect
                jet_effect = 15 * np.exp(-((lat - 45) / 15)**2) if level < 300 else 0
                
                for j, lon in enumerate(lons):
                    wind_data[t, k, i, j] = (
                        base_wind + jet_effect + 
                        np.random.normal(0, 3) +
                        5 * np.cos(np.radians(lat)) * np.sin(np.radians(lon))
                    )
    
    # Ensure no negative wind speeds
    wind_data = np.maximum(wind_data, 0)
    
    # Create xarray dataset
    ds = xr.Dataset(
        {
            'wind_speed': (['time', 'level', 'latitude', 'longitude'], wind_data)
        },
        coords={
            'time': time,
            'level': levels,
            'latitude': lats,
            'longitude': lons
        },
        attrs={
            'title': 'Dummy 3D Wind Speed Dataset',
            'description': 'Synthetic wind speed data with vertical levels'
        }
    )
    
    ds.wind_speed.attrs = {
        'units': 'm/s',
        'long_name': 'Wind Speed',
        'standard_name': 'wind_speed'
    }
    
    ds.level.attrs = {
        'units': 'hPa',
        'long_name': 'Pressure Level',
        'positive': 'down'
    }
    
    print(f"✓ 3D dataset created: {ds.sizes}")
    return ds


def demonstrate_plotting():
    """Demonstrate the plotting functionality with different datasets."""
    print("\n" + "="*60)
    print("CLIMATRIX INTERACTIVE PLOTTING DEMONSTRATION")
    print("="*60)
    
    # Check if plotting dependencies are available
    try:
        import plotly
        import dash
        print("✓ Plotting dependencies (plotly, dash) are available")
    except ImportError as e:
        print(f"✗ Plotting dependencies not available: {e}")
        print("Install with: pip install plotly dash")
        return
    
    # Create example datasets
    dense_ds = create_dummy_dense_dataset()
    sparse_ds = create_dummy_sparse_dataset()
    volume_ds = create_dummy_3d_dataset()
    
    print("\n" + "-"*50)
    print("EXAMPLE 1: Dense Temperature Dataset")
    print("-"*50)
    print("This example shows a global temperature dataset with:")
    print("- Daily time series (365 days)")
    print("- Global spatial coverage (50x100 grid)")
    print("- Time slider for animation")
    print("- 2D/3D view toggle")
    
    try:
        # Convert to climatrix dataset
        cm_dense = dense_ds.cm
        
        # Create interactive plot
        print("\nCreating interactive plot for dense dataset...")
        plot_dense = cm.plot.Plot(cm_dense, port=8050, auto_open=False)
        
        print("✓ Dense dataset plot created successfully")
        print("To view: plot_dense.show()  # This will open in browser on port 8050")
        
        # Save static version
        print("Saving static HTML version...")
        plot_dense.save_html("/tmp/dense_temperature_plot.html")
        print("✓ Static plot saved to /tmp/dense_temperature_plot.html")
        
    except Exception as e:
        print(f"✗ Error creating dense dataset plot: {e}")
    
    print("\n" + "-"*50)
    print("EXAMPLE 2: Sparse Station Dataset")
    print("-"*50)
    print("This example shows station precipitation data with:")
    print("- 50 weather stations globally")
    print("- Daily precipitation measurements")
    print("- Scatter plot visualization")
    
    try:
        # Convert to climatrix dataset
        cm_sparse = sparse_ds.cm
        
        # Create interactive plot
        print("\nCreating interactive plot for sparse dataset...")
        plot_sparse = cm.plot.Plot(cm_sparse, port=8051, auto_open=False)
        
        print("✓ Sparse dataset plot created successfully")
        print("To view: plot_sparse.show()  # This will open in browser on port 8051")
        
        # Save static version
        plot_sparse.save_html("/tmp/sparse_precipitation_plot.html")
        print("✓ Static plot saved to /tmp/sparse_precipitation_plot.html")
        
    except Exception as e:
        print(f"✗ Error creating sparse dataset plot: {e}")
    
    print("\n" + "-"*50)
    print("EXAMPLE 3: 3D Wind Speed Dataset")
    print("-"*50)
    print("This example shows 3D atmospheric data with:")
    print("- Multiple pressure levels (7 levels)")
    print("- Time series (7 days)")
    print("- Vertical level slider")
    print("- Time animation")
    
    try:
        # Convert to climatrix dataset
        cm_volume = volume_ds.cm
        
        # Create interactive plot
        print("\nCreating interactive plot for 3D dataset...")
        plot_volume = cm.plot.Plot(cm_volume, port=8052, auto_open=False)
        
        print("✓ 3D dataset plot created successfully")
        print("To view: plot_volume.show()  # This will open in browser on port 8052")
        
        # Save static version
        plot_volume.save_html("/tmp/wind_speed_3d_plot.html")
        print("✓ Static plot saved to /tmp/wind_speed_3d_plot.html")
        
    except Exception as e:
        print(f"✗ Error creating 3D dataset plot: {e}")
    
    print("\n" + "="*60)
    print("INTERACTIVE PLOTTING FEATURES")
    print("="*60)
    print("Each plot includes the following interactive features:")
    print("• Time slider (if time dimension present)")
    print("• Vertical level slider (if vertical dimension present)")
    print("• 2D flat view / 3D globe view toggle")
    print("• Zoom and pan capabilities")
    print("• Click and drag to select regions")
    print("• Loading indicators during updates")
    print("• Error handling with user notifications")
    print("• Material design interface")
    print("• Appropriate visualization for sparse vs dense data")
    
    print("\n" + "="*60)
    print("USAGE INSTRUCTIONS")
    print("="*60)
    print("To run these examples interactively:")
    print("1. Make sure plotting dependencies are installed:")
    print("   pip install plotly dash")
    print("2. Run the individual plot commands:")
    print("   plot_dense.show()   # Dense temperature data")
    print("   plot_sparse.show()  # Sparse station data")
    print("   plot_volume.show()  # 3D atmospheric data")
    print("3. Open the provided URLs in your browser")
    print("4. Use the controls panel to interact with the data")
    print("5. Press Ctrl+C in terminal to stop the server")
    
    return {
        'dense': plot_dense if 'plot_dense' in locals() else None,
        'sparse': plot_sparse if 'plot_sparse' in locals() else None,
        'volume': plot_volume if 'plot_volume' in locals() else None
    }


if __name__ == "__main__":
    plots = demonstrate_plotting()
    
    print("\n" + "="*60)
    print("READY TO INTERACT!")
    print("="*60)
    print("Example plots have been created and are ready to use.")
    print("Static HTML versions saved to /tmp/ for offline viewing.")
    print("\nTo start interactive sessions, run:")
    for name, plot in plots.items():
        if plot:
            print(f"  plots['{name}'].show()")
    print("\nEnjoy exploring your climate data interactively!")