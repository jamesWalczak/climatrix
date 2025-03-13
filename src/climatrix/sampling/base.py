from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from climatrix.dataset.base import BaseDataset
    from climatrix.dataset.dense import SparseDataset




class BaseSampler(ABC):
    __slots__ = ("dataset", "portion", "number")

    dataset: BaseDataset
    portion: float | None
    number: int | None

    def __init__(self, dataset: BaseDataset, portion: float | None = None, number: int | None = None):
        self.dataset = dataset
        if not (portion or number):
            raise ValueError("Either portion or number must be provided")
        if portion and number:
            raise ValueError("Either portion or number must be provided, but not both")
        self.portion = portion
        self.number = number

    def get_all_lats(self):
        return self.dataset.latitude.values
    
    def get_all_lons(self):
        return self.dataset.longitude.values
    
    def get_sample_size(self):
        if self.portion:
            n = int(self.portion * len(self.get_all_lats()) * len(self.get_all_lons()))
        else:
            n = self.number
            if n > len(self.get_all_lats()) * len(self.get_all_lons()):
                raise ValueError("Number of samples is too large")
        return n        

    @abstractmethod
    def sample(self) -> SparseDataset:
        raise NotImplementedError
