from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

from climatrix.dataset.domain import Domain
from climatrix.utils.hyperparameter import Hyperparameter

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
    
    # Class registry for reconstruction methods
    _registry: ClassVar[dict[str, type[BaseReconstructor]]] = {}

    dataset: BaseClimatrixDataset

    def __init__(
        self, dataset: BaseClimatrixDataset, target_domain: Domain
    ) -> None:
        self.dataset = dataset
        self.target_domain = target_domain
        self._validate_types(dataset, target_domain)

    def __init_subclass__(cls, **kwargs):
        """Register subclasses automatically."""
        super().__init_subclass__(**kwargs)
        # Register the class with a lowercase name derived from the class name
        name = cls.__name__.lower().replace('reconstructor', '')
        cls._registry[name] = cls

    @classmethod
    def get(cls, method: str) -> type[BaseReconstructor]:
        """
        Get a reconstruction class by method name.
        
        Parameters
        ----------
        method : str
            The reconstruction method name (e.g., 'idw', 'ok', 'sinet', 'siren').
            
        Returns
        -------
        type[BaseReconstructor]
            The reconstruction class.
            
        Raises
        ------
        ValueError
            If the method is not supported.
        """
        method = method.lower().strip()
        if method not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(f"Unknown method '{method}'. Available methods: {available}")
        return cls._registry[method]

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

    @classmethod
    def hparams(cls) -> dict[str, dict[str, any]]:
        """
        Get hyperparameter definitions from Hyperparameter descriptors.
        
        Returns
        -------
        dict[str, dict[str, any]]
            Dictionary mapping parameter names to their definitions.
            Each parameter definition contains:
            - 'type': the parameter type
            - 'bounds': tuple of (min, max) for numeric parameters (if defined)
            - 'values': list of valid values for categorical parameters (if defined)
            - 'default': default value (if defined)
        """
        result = {}
        
        # Look through the class attributes for Hyperparameter descriptors
        for attr_name in dir(cls):
            attr_value = getattr(cls, attr_name)
            if isinstance(attr_value, Hyperparameter):
                result[attr_name] = attr_value.get_spec()
                    
        return result
