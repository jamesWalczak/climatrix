from abc import ABC, abstractmethod

from climatrix.dataset.dense import DenseDataset
from climatrix.dataset.sparse import SparseDataset


class BaseReconstructor(ABC):
    __slots__ = ("dataset",)
    dataset: SparseDataset

    def __init__(self, dataset: SparseDataset):
        if not isinstance(dataset, SparseDataset):
            raise TypeError("Only SparseDataset object can be reconstructed")
        self.dataset = dataset

    @abstractmethod
    def reconstruct(self) -> DenseDataset:
        raise NotImplementedError
