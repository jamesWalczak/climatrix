import numpy as np
import xarray as xr

from climatrix.dataset.axis import Axis
from climatrix.dataset.sparse import (
    DynamicSparseDataset,
    StaticSparseDataset,
)
from climatrix.sampling.base import BaseSampler


class UniformSampler(BaseSampler):

    def _sample_data(self, n):
        selected_lats = np.random.choice(self.dataset.latitude, n)
        selected_lons = np.random.choice(self.dataset.longitude, n)
        data = self.dataset.da.sel(
            {
                self.dataset.latitude_name: xr.DataArray(
                    selected_lats, dims=[Axis.POINT]
                ),
                self.dataset.longitude_name: xr.DataArray(
                    selected_lons, dims=[Axis.POINT]
                ),
            },
            method="nearest",
        )
        return (
            DynamicSparseDataset(data)
            if self.dataset.time_name and self.dataset.time.size > 1
            else StaticSparseDataset(data)
        )

    def _sample_no_nans(self, n):
        if self.dataset.is_dynamic:
            stack_keys = {
                Axis.POINT: [
                    self.dataset.time_name,
                    self.dataset.latitude_name,
                    self.dataset.longitude_name,
                ]
            }
        else:
            stack_keys = {
                Axis.POINT: [
                    self.dataset.latitude_name,
                    self.dataset.longitude_name,
                ]
            }
        stacked = self.dataset.da.stack(**stack_keys)
        valid_da = stacked[stacked.notnull().compute()]
        rand_idx = np.arange(len(valid_da))
        np.random.shuffle(rand_idx)
        rand_idx = rand_idx[:n]
        sparse_data = valid_da.isel({Axis.POINT: rand_idx})
        selected_lats = sparse_data[self.dataset.latitude_name].values
        selected_lons = sparse_data[self.dataset.longitude_name].values

        data = self.dataset.da.sel(
            {
                self.dataset.latitude_name: xr.DataArray(
                    selected_lats, dims=[Axis.POINT]
                ),
                self.dataset.longitude_name: xr.DataArray(
                    selected_lons, dims=[Axis.POINT]
                ),
            },
            method="nearest",
        )
        return (
            DynamicSparseDataset(data)
            if self.dataset.time_name and self.dataset.time.size > 1
            else StaticSparseDataset(data)
        )
