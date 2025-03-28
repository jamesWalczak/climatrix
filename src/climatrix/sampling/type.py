from enum import Enum
from typing import Self

from climatrix.sampling.normal import NormalSampler
from climatrix.sampling.uniform import UniformSampler


class SamplingType(Enum):
    UNIFORM = UniformSampler
    NORMAL = NormalSampler

    def __missing__(self, value):
        raise ValueError(f"Unknown sampling type: {value}")

    @classmethod
    def get(cls, value: str | Self):
        if isinstance(value, cls):
            return value
        try:
            return cls[value.upper()]
        except KeyError:
            raise ValueError(f"Unknown sampling type: {value}")
