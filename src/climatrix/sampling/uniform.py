import numpy as np
import xarray as xr

from climatrix.dataset.sparse import (
    SPARSE_DIM,
    DynamicSparseDataset,
    StaticSparseDataset,
)
from climatrix.sampling.base import BaseSampler


class UniformSampler(BaseSampler):

    def _sample_data(self, lats, lons, n):
        selected_lats = np.random.choice(lats, n)
        selected_lons = np.random.choice(lons, n)
        data = self.dataset.da.sel(
            {
                self.dataset.latitude_name: xr.DataArray(
                    selected_lats, dims=[SPARSE_DIM]
                ),
                self.dataset.longitude_name: xr.DataArray(
                    selected_lons, dims=[SPARSE_DIM]
                ),
            },
            method="nearest",
        )
        return (
            DynamicSparseDataset(data)
            if self.dataset.time_name
            else StaticSparseDataset(data)
        )

    def _sample_no_nans(self, lats, lons, n):
        stacked = self.dataset.da.stack(
            **{
                SPARSE_DIM: [
                    self.dataset._def.time_name,
                    self.dataset._def.latitude_name,
                    self.dataset._def.longitude_name,
                ]
            }
        )
        valid_da = stacked[stacked.notnull().compute()]
        rand_idx = np.arange(len(valid_da))
        np.random.shuffle(rand_idx)
        rand_idx = rand_idx[:n]
        sparse_data = valid_da.isel({SparseDataset.SPARSE_DIM: rand_idx})
        # return SparseDataset(sparse_data, self.dataset._def)
        selected_lats = sparse_data[self.dataset._def.latitude_name].values
        selected_lons = sparse_data[self.dataset._def.longitude_name].values

        data = self.dataset.da.sel(
            {
                self.dataset._def.latitude_name: xr.DataArray(
                    selected_lats, dims=[SparseDataset.SPARSE_DIM]
                ),
                self.dataset._def.longitude_name: xr.DataArray(
                    selected_lons, dims=[SparseDataset.SPARSE_DIM]
                ),
            },
            method="nearest",
        )
        return SparseDataset(data, self.dataset._def)
