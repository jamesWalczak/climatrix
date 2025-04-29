import numpy as np
import xarray as xr

import climatrix as cm

dense_domain = cm.Domain.from_lat_lon(
    slice(-10, 20, 0.1), slice(-40, 30, 0.2), kind="dense"
)
sparse_domain = cm.Domain.from_lat_lon(
    np.linspace(0, 100, 10), np.linspace(0, 100, 10), kind="sparse"
)
