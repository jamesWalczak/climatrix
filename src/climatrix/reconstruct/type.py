from enum import Enum
from typing import Self

from climatrix.reconstruct.idw import IDWReconstructor


class ReconstructionType(Enum):
    IDW = IDWReconstructor

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
