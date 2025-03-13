import numpy as np
import xarray as xr

from climatrix.sampling.base import BaseSampler


class UniformSampler(BaseSampler):



    def sample(self):
        from climatrix.dataset.sparse import SparseDataset

        n = self.get_sample_size()
        lats = self.get_all_lats()
        lons = self.get_all_lons()

        selected_lats = np.random.choice(lats, n)
        selected_lons = np.random.choice(lons, n)

        res = self.dataset.dset.sel({self.dataset._def.latitude_name: xr.DataArray(selected_lats, dims=[SparseDataset.SPARSE_DIM]), 
            self.dataset._def.longitude_name: xr.DataArray(selected_lons, dims=[SparseDataset.SPARSE_DIM])}, method='nearest')        

        return SparseDataset(res, self.dataset._def)