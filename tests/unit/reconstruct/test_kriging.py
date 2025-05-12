import pytest

from climatrix.reconstruct.kriging import OrdinaryKrigingReconstructor

from .test_base_interface import TestBaseReconstructor


class TestKrigingReconstructor(TestBaseReconstructor):
    __test__ = True

    @pytest.fixture
    def reconstructor_class(self):
        return OrdinaryKrigingReconstructor
