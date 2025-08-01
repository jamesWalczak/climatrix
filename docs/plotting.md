# 📈 Interactive Plotting

Climatrix provides a powerful interactive plotting utility that creates beautiful, web-based visualizations of your climate data using Plotly and Dash.

## 🚀 Quick Start

```python
import climatrix as cm
import xarray as xr

# Load your climate dataset
ds = xr.open_dataset("your_data.nc")
cm_ds = ds.cm

# Create and launch interactive plot
plot = cm.plot.Plot(cm_ds)
plot.show()  # Opens in browser
```

## ✨ Features

### 🎨 Material Design Interface
- Clean, professional styling with responsive controls
- Intuitive layout with controls panel and main visualization area
- Loading indicators for smooth user experience

### ⏰ Time Animation
- Interactive time slider for datasets with temporal dimensions
- Smooth animation through time series data
- Customizable time step navigation

### 📏 Vertical Navigation
- Slider control for atmospheric/oceanic layers
- Navigate through pressure levels, depth layers, or other vertical coordinates
- Real-time updates as you change vertical position

### 🌍 Dual View Modes
- **2D Flat View**: Traditional map projections with zoom and pan
- **3D Globe View**: Interactive spherical visualization for global datasets

### 🔍 Smart Data Visualization
- **Sparse Data**: Automatically renders as scatter plots on maps
- **Dense Data**: Displays as heatmaps or contour plots
- **Threshold**: Datasets with < 1000 spatial points treated as sparse

### ⚡ Performance Optimized
- Efficient rendering for large datasets
- Lazy loading of data slices
- Optimized memory usage

### 🖱️ Interactive Controls
- Zoom and pan functionality
- Click-to-select regions
- Drag to create selection boxes
- Hover tooltips with data values

## 📦 Installation

Install the plotting dependencies:

```bash
# Install with plotting extras
pip install climatrix[plot]

# Or install manually
pip install plotly dash
```

## 🛠️ Advanced Usage

### Custom Configuration

```python
# Customize the plot server
plot = cm.plot.Plot(
    dataset=cm_ds,
    port=8050,          # Custom port
    host='0.0.0.0',     # Make accessible on network
    auto_open=False,    # Don't auto-open browser
    debug=True          # Enable debug mode
)
```

### Export Static Plots

```python
# Save as standalone HTML file
plot.save_html("my_climate_plot.html")
```

### Multiple Datasets

```python
# Create multiple plots on different ports
temp_plot = cm.plot.Plot(temperature_data, port=8050)
precip_plot = cm.plot.Plot(precipitation_data, port=8051)
wind_plot = cm.plot.Plot(wind_data, port=8052)

# Launch all plots
temp_plot.show()   # Will open first plot
precip_plot.show() # Will open second plot (different port)
wind_plot.show()   # Will open third plot
```

## 📊 Data Types Supported

### Dense Gridded Data
- Regular latitude/longitude grids
- Displayed as heatmaps or contour plots
- Supports time series animation
- Example: Global temperature fields, precipitation grids

### Sparse Point Data
- Weather station data
- Buoy measurements
- Satellite retrievals at specific locations
- Displayed as scatter plots on maps
- Example: Station observations, ship tracks

### 3D Atmospheric/Oceanic Data
- Multiple vertical levels (pressure, depth, height)
- Time series with vertical structure
- Vertical slider for level navigation
- Example: Atmospheric profiles, ocean temperature/salinity

### Time Series Data
- Temporal datasets with regular or irregular time steps
- Time slider for animation
- Supports daily, monthly, yearly data
- Example: Climate model output, observational records

## 🎯 Examples

### Example 1: Global Temperature Data

```python
import climatrix as cm
import xarray as xr
import numpy as np

# Create example temperature dataset
lats = np.linspace(-90, 90, 50)
lons = np.linspace(-180, 180, 100)
time = np.arange(np.datetime64('2020-01-01'), 
                 np.datetime64('2020-12-31'), 
                 np.timedelta64(1, 'D'))

# Generate synthetic temperature data
temp_data = np.random.normal(15, 10, (len(time), len(lats), len(lons)))

ds = xr.Dataset(
    {'temperature': (['time', 'latitude', 'longitude'], temp_data)},
    coords={'time': time, 'latitude': lats, 'longitude': lons}
)

# Create interactive plot
plot = cm.plot.Plot(ds.cm)
plot.show()
```

### Example 2: Weather Station Data

```python
# Create sparse station dataset
n_stations = 50
station_lats = np.random.uniform(-90, 90, n_stations)
station_lons = np.random.uniform(-180, 180, n_stations)
precip_data = np.random.exponential(5, n_stations)

ds = xr.Dataset(
    {'precipitation': (['station'], precip_data)},
    coords={
        'station': [f"STN_{i:03d}" for i in range(n_stations)],
        'latitude': ('station', station_lats),
        'longitude': ('station', station_lons)
    }
)

# Create scatter plot
plot = cm.plot.Plot(ds.cm)
plot.show()
```

### Example 3: 3D Atmospheric Data

```python
# Create 3D dataset with vertical levels
levels = [1000, 850, 700, 500, 300, 200, 100]  # Pressure levels
wind_data = np.random.normal(20, 10, (len(time), len(levels), len(lats), len(lons)))

ds = xr.Dataset(
    {'wind_speed': (['time', 'level', 'latitude', 'longitude'], wind_data)},
    coords={
        'time': time,
        'level': levels,
        'latitude': lats,
        'longitude': lons
    }
)

# Create 3D plot with vertical navigation
plot = cm.plot.Plot(ds.cm)
plot.show()
```

## 🔧 Troubleshooting

### Common Issues

**Dependencies Not Found**
```
MissingDependencyError: Interactive plotting requires plotly and dash.
```
Solution: Install plotting dependencies with `pip install plotly dash`

**Port Already in Use**
```
OSError: [Errno 98] Address already in use
```
Solution: Use a different port with `Plot(dataset, port=8051)`

**Memory Issues with Large Datasets**
- Try subsetting data first: `ds.sel(time=slice('2020-01-01', '2020-01-31'))`
- Use sparse sampling: `ds.thin(latitude=2, longitude=2)`

### Performance Tips

1. **Subset Large Datasets**: Use `ds.sel()` or `ds.isel()` to reduce data size
2. **Use Appropriate Data Types**: Convert to float32 if precision allows
3. **Optimize Time Ranges**: Start with shorter time periods for initial exploration
4. **Consider Spatial Resolution**: Downsample high-resolution grids for faster rendering

## 🎨 Customization

### Styling Options
The plot interface uses Material Design principles with customizable colors and themes. Future versions will support additional styling options.

### Extending Functionality
The plotting module is designed to be extensible. Advanced users can subclass the `Plot` class to add custom visualization types or modify the default behavior.

## 🚀 Demo Script

Run the complete demonstration script to see all features:

```bash
python examples/interactive_plotting_demo.py
```

This script creates sample datasets and demonstrates:
- Dense temperature data with time animation
- Sparse precipitation station data
- 3D atmospheric wind data with vertical levels
- All interactive features and controls

The demo includes both live interactive plots and static HTML exports for offline viewing.