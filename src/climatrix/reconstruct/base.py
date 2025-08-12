from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

from climatrix.dataset.domain import Domain
from climatrix.optim.hyperparameter import Hyperparameter

if TYPE_CHECKING:
    from climatrix.dataset.base import BaseClimatrixDataset

log = logging.getLogger(__name__)


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

    __slots__ = ("dataset", "target_domain")

    # Class registry for reconstruction methods
    _registry: ClassVar[dict[str, type[BaseReconstructor]]] = {}
    NAME: ClassVar[str] = "<not set>"
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
        cls._registry[cls.NAME] = cls

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

        Notes
        -----
        The `method` parameter should reflect the `NAME` class attribute
        of the selected reconstructor class.
        """
        method = method.lower().strip()
        if method not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Unknown method '{method}'. Available methods: {available}"
            )
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
    def get_hparams(cls) -> dict[str, dict[str, any]]:
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

        for attr_name in dir(cls):
            attr_value = getattr(cls, attr_name)
            if isinstance(attr_value, Hyperparameter):
                result[attr_name] = attr_value.get_spec()

        return result

    @classmethod
    def get_available_methods(cls) -> list[str]:
        """
        Get a list of available reconstruction methods.

        Returns
        -------
        list[str]
            List of method names (e.g., 'idw', 'ok', 'sinet', 'siren').
        """
        return list(cls._registry.keys())

    @classmethod
    def update_bounds(
        cls, bounds: dict | None = None, values: dict | None = None
    ) -> None:
        """
        Update the bounds of hyperparameters in the class.

        If bound is defined as tuple, it represents a range (min, max).
        If as a list, it represents a set of valid values.

        Parameters
        ----------
        **bounds : dict[str, tuple]
            Keyword arguments where keys are hyperparameter names
            and values are tuples defining new bounds.
        """
        if bounds is None:
            bounds = {}
        if values is None:
            values = {}
        if bounds is None and values is None:
            return
        if not isinstance(bounds, dict) or not isinstance(values, dict):
            raise TypeError(
                "bounds and values must be dictionaries mapping parameter names to tuples or lists"
            )
        hparams = cls.get_hparams()
        for param_name, new_bounds in bounds.items():
            if param_name not in hparams:
                continue
            hparam = getattr(cls, param_name, None)
            if hparam is not None:
                log.debug(
                    "Updating bounds for parameter '%s' to %s",
                    param_name,
                    new_bounds,
                )
                hparam.bounds = new_bounds
        for param_name, new_values in values.items():
            if param_name not in hparams:
                continue
            hparam = getattr(cls, param_name, None)
            if hparam is not None:
                log.debug(
                    "Updating values for parameter '%s' to %s",
                    param_name,
                    new_values,
                )
                hparam.values = new_values
