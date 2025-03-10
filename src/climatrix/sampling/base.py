from abc import ABC, abstractmethod

from climatrix.dataset.base import BaseDataset
from climatrix.dataset.dense import SparseDataset


class BaseSampler(ABC):
    __slots__ = "dataset"

    dataset: BaseDataset

    def __init__(self, dataset: BaseDataset):
        self.dataset = dataset

    @abstractmethod
    def sample(self) -> SparseDataset:
        raise NotImplementedError
