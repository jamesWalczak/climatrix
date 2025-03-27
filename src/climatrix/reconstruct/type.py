from enum import Enum
from typing import Self

from climatrix.reconstruct.idw import IDWReconstructor
from climatrix.reconstruct.kriging import OrdinaryKrigingReconstructor
from climatrix.reconstruct.siren.siren import SIRENReconstructor


class ReconstructionType(Enum):
    IDW = IDWReconstructor
    OK = OrdinaryKrigingReconstructor
    SIREN = SIRENReconstructor

    def __missing__(self, value):
        raise ValueError(f"Unknown reconstruction method: {value}")

    @classmethod
    def get(cls, value: str | Self):
        if isinstance(value, cls):
            return value
        try:
            return cls[value.upper()]
        except KeyError:
            raise ValueError(f"Unknown reconstruction type: {value}")
