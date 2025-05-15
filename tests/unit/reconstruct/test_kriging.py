import importlib

import pytest

from climatrix.reconstruct.kriging import OrdinaryKrigingReconstructor

from .test_base_interface import TestBaseReconstructor


@pytest.mark.skipif(
    not importlib.util.find_spec("pykrige"), reason="pykrige is not installed"
)
class TestKrigingReconstructor(TestBaseReconstructor):
    __test__ = True

    @pytest.fixture
    def reconstructor_class(self):
        return OrdinaryKrigingReconstructor
