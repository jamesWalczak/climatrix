from climatrix.dataset.base import BaseDataset

dataset = BaseDataset.load("/storage/tul/projects/climatrix/data/era5-land.nc")
europe = dataset.subset(north=71, south=36, west=-24, east=35)
europe.plot()