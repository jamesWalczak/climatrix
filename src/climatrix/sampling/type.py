from enum import Enum

from climatrix.sampling.uniform import UniformSampler


class SamplingType(Enum):
    UNIFORM = UniformSampler