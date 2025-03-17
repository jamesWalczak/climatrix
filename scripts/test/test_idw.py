from climatrix.dataset.base import BaseDataset
from climatrix.dataset.dense import DenseDataset

dataset: DenseDataset = BaseDataset.load(
    "/storage/tul/projects/climatrix/data/era5-land.nc"
)
europe = dataset.subset(north=71, south=36, west=-24, east=35)
sparse = europe.sample(number=1000, kind="uniform", nan_policy="resample")
sparse.reconstruct(
    lat=slice(71, 36, -0.1), lon=slice(-24, 35, 0.1), method="idw", k=10
).plot()
