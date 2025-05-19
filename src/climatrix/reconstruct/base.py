from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

from climatrix.dataset.domain import Domain

if TYPE_CHECKING:
    from climatrix.dataset.base import BaseClimatrixDataset


class BaseReconstructor(ABC):
    """
    Base class for all dataset reconstruction methods.

    Attributes
    ----------
    dataset : BaseClimatrixDataset
        The dataset to be reconstructed.
    target_domain : Domain
        The target domain for the reconstruction.

    """

    __slots__ = ("dataset", "query_lat", "query_lon")

    dataset: BaseClimatrixDataset

    def __init__(
        self, dataset: BaseClimatrixDataset, target_domain: Domain
    ) -> None:
        self.dataset = dataset
        self.target_domain = target_domain
        self._validate_types(dataset, target_domain)

    def _validate_types(self, dataset, domain: Domain) -> None:
        from climatrix.dataset.base import BaseClimatrixDataset

        if not isinstance(dataset, BaseClimatrixDataset):
            raise TypeError("dataset must be a BaseClimatrixDataset object")

        if not isinstance(domain, Domain):
            raise TypeError("domain must be a Domain object")

    @abstractmethod
    def reconstruct(self) -> BaseClimatrixDataset:
        """
        Reconstruct the dataset using the specified method.

        This is an abstract method that must be implemented
        by subclasses.

        The data are reconstructed for the target domain, passed
        in the initializer.

        Returns
        -------
        BaseClimatrixDataset
            The reconstructed dataset.
        """
        raise NotImplementedError
