from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, ClassVar, Self

from climatrix.exceptions import TooLargeSamplePortionWarning

if TYPE_CHECKING:
    from climatrix.dataset.dense import DenseDataset, SparseDataset


class NaNPolicy(Enum):
    IGNORE = "ignore"
    RESAMPLE = "resample"
    RAISE = "raise"

    def __missing__(self, value):
        raise ValueError(f"Unknown NaN policy: {value}")

    @classmethod
    def get(cls, value: str | Self):
        if isinstance(value, cls):
            return value
        try:
            return cls[value.upper()]
        except KeyError:
            raise ValueError(f"Unknown Nan policy: {value}")


class BaseSampler(ABC):
    __slots__ = ("dataset", "portion", "number")
    TOO_MANY_VALUES: ClassVar[float] = 1e8

    dataset: DenseDataset
    portion: float | None
    number: int | None

    def __init__(
        self,
        dataset: DenseDataset | SparseDataset,
        portion: float | None = None,
        number: int | None = None,
    ):
        from climatrix.dataset.dense import DenseDataset
        from climatrix.dataset.sparse import SparseDataset

        self.dataset = dataset
        if not isinstance(dataset, (DenseDataset, SparseDataset)):
            raise TypeError(
                f"Dataset must be of type DenseDataset or SparseDataset, "
                f"not {type(dataset)}"
            )
        if not (portion or number):
            raise ValueError("Either portion or number must be provided")
        if portion and number:
            raise ValueError(
                "Either portion or number must be provided, but not both"
            )
        if (portion and portion > 1.0) or (
            number and number > self.dataset.size
        ):
            warnings.warn(
                "Requesting more than 100% of the data will result in "
                "duplicates and excessive memory usage",
                TooLargeSamplePortionWarning,
            )
        self.portion = portion
        self.number = number

    def get_all_lats(self):
        return self.dataset.latitude.values

    def get_all_lons(self):
        return self.dataset.longitude.values

    def get_sample_size(self):
        if self.portion:
            n = int(
                self.portion
                * len(self.get_all_lats())
                * len(self.get_all_lons())
            )
        else:
            n = self.number
            if n > len(self.get_all_lats()) * len(self.get_all_lons()):
                raise ValueError("Number of samples is too large")
        return n

    @abstractmethod
    def _sample_data(self, n) -> SparseDataset:
        raise NotImplementedError

    @abstractmethod
    def _sample_no_nans(self, n) -> SparseDataset:
        raise NotImplementedError

    def _ensure_dataset_not_too_large(self):
        import math

        size = math.prod(self.dataset.da.sizes.values())
        if size > self.TOO_MANY_VALUES:
            raise ValueError(
                "Dataset is too large. The limit is "
                f"{self.TOO_MANY_VALUES}, but the dataset has "
                f"{size} values. Subset or resample the dataset "
                "before sampling."
            )

    def sample(self, nan_policy: NaNPolicy | str = "ignore") -> SparseDataset:
        nan_policy = NaNPolicy.get(nan_policy)
        n = self.get_sample_size()
        lats = self.get_all_lats()
        lons = self.get_all_lons()

        if n == self.dataset.size:
            return SparseDataset(self.dataset.da)

        match nan_policy:
            case NaNPolicy.IGNORE:
                return self._sample_data(lats, lons, n)
            case NaNPolicy.RESAMPLE:
                self._ensure_dataset_not_too_large()
                return self._sample_no_nans(lats, lons, n)
            case NaNPolicy.RAISE:
                res = self._sample_data(lats, lons, n)
                if res.da.isnull().any():
                    raise ValueError("Not all points have data")
                return res
            case _:
                raise ValueError(f"Unknown NaNPolicy: {nan_policy}")
