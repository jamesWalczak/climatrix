from climatrix.dataset.base import BaseDataset
from climatrix.dataset.dense import DenseDataset

dataset: DenseDataset = BaseDataset.load("/storage/tul/projects/climatrix/data/era5-land.nc")
europe = dataset.subset(north=71, south=36, west=-24, east=35)
europe.sample(number=1000, kind="normal", nan_policy="ignore", sigma=3, center_point=(19, 51)).plot()