import numpy as np
import pyproj
import rioxarray
from scipy import ndimage

# 1. Load DSM raster (assumed to be in EPSG:3035)
dsm_path = "/home/jakub/Downloads/copernicus_DSM_1000m_mood_bbox_epsg3035.tif"
dsm = rioxarray.open_rasterio(
    dsm_path, masked=True
).squeeze()  # remove band dim if any

# 2. Calculate slope (degrees) from DSM elevation data
# Using simple finite differences (gradient) with spacing from CRS units (meters)
# slope = arctan(sqrt(dz/dx^2 + dz/dy^2)) in degrees

# Get pixel size in meters (assume square pixels)
res_x = abs(dsm.rio.resolution()[0])
res_y = abs(dsm.rio.resolution()[1])

dsm.values = dsm.values / 1_000

# Calculate gradients in x and y direction
dz_dx = ndimage.sobel(dsm.values, axis=1) / (8 * res_x)  # axis=1 is x
dz_dy = ndimage.sobel(dsm.values, axis=0) / (8 * res_y)  # axis=0 is y

# Calculate slope in radians, then convert to degrees
slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
slope_deg = np.degrees(slope_rad)

# Mask slope where DSM is masked or NaN
valid_mask = ~np.isnan(dsm.values) & (dsm.values != dsm.rio.nodata)

# 3. Prepare coordinate transformer (EPSG:3035 -> EPSG:4326)
transformer = pyproj.Transformer.from_crs(
    dsm.rio.crs, "EPSG:4326", always_xy=True
)

# Extract x/y coordinates (meters)
x = dsm.coords["x"].values
y = dsm.coords["y"].values

# Create meshgrid of pixel centers
xx, yy = np.meshgrid(x, y)

# Flatten arrays
xx_flat = xx.flatten()
yy_flat = yy.flatten()
slope_flat = slope_deg.flatten()
elevation_flat = dsm.values.flatten()
valid_flat = valid_mask.flatten()

# Filter valid points
xx_valid = xx_flat[valid_flat]
yy_valid = yy_flat[valid_flat]
slope_valid = slope_flat[valid_flat]
elevation_valid = elevation_flat[valid_flat]

# 4. Transform valid coordinates to lat/lon degrees
lon_valid, lat_valid = transformer.transform(xx_valid, yy_valid)

# 5. Stack slope_results into Nx3 array: [lat, lon, slope]
slope_result = np.vstack([lat_valid, lon_valid, slope_valid]).T
elevation_result = np.vstack([lat_valid, lon_valid, elevation_valid]).T

print(f"Valid points count: {slope_result.shape[0]}")
print("Sample points (lat, lon, slope):")

# 6. Save slope_result as numpy array
mask = np.all(
    [np.isfinite(slope_result[:, -1]), slope_result[:, -1] >= 0], axis=0
)
slope_result = slope_result[mask]
elevation_result = elevation_result[mask]

np.save("lat_lon_elevation.npy", elevation_result)
np.save("lat_lon_slope.npy", slope_result)
