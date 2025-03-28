from enum import Enum
from typing import Self

from climatrix.reconstruct.idw import IDWReconstructor
from climatrix.reconstruct.kriging import OrdinaryKrigingReconstructor
from climatrix.reconstruct.siren.siren import SIRENReconstructor


class ReconstructionType(Enum):
    """The available reconstruction types."""

    IDW = IDWReconstructor
    OK = OrdinaryKrigingReconstructor
    SIREN = SIRENReconstructor

    def __missing__(self, value):
        raise ValueError(f"Unknown reconstruction method: {value}")

    @classmethod
    def get(cls, value: str | Self):
        """
        Get the reconstruction type given by `value`.

        If `value` is an instance of ReconstructionType,
        return it as is.
        If `value` is a string, return the corresponding
        ReconstructionType.
        If `value` is neither an instance of ReconstructionType
        nor a string,
        raise a ValueError.

        Parameters
        ----------
        value : str | ReconstructionType
            The reconstruction type to get.

        Returns
        -------
        ReconstructionType
            The reconstruction type.

        Raises
        ------
        ValueError
            If `value` is not a valid reconstruction type.
        """
        if isinstance(value, cls):
            return value
        try:
            return cls[value.upper()]
        except KeyError:
            raise ValueError(f"Unknown reconstruction type: {value}")
