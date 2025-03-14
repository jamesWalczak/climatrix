from climatrix.dataset.base import BaseDataset

dataset = BaseDataset.load("/storage/tul/projects/climatrix/data/era5-land.nc")
part = dataset.subset(west=180, east=360)
part.plot()