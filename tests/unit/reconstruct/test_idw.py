import pytest

from climatrix.reconstruct.idw import IDWReconstructor
from tests.unit.utils import skip_on_error

from .test_base_interface import TestBaseReconstructor, parametrize_all


class TestIDWReconstructor(TestBaseReconstructor):
    __test__ = True

    @pytest.fixture
    def reconstructor_class(self):
        return IDWReconstructor

    @parametrize_all()
    @skip_on_error(NotImplementedError)
    def test_raise_on_k_min_negative(self, dataset):
        with pytest.raises(ValueError, match="k must be >= 1"):
            IDWReconstructor(dataset, dataset.domain, k_min=-1, k=-1)

    @parametrize_all()
    @skip_on_error(NotImplementedError)
    def test_raise_on_k_min_bigger_than_k_negative(self, dataset):
        with pytest.raises(ValueError, match="k_min must be <= k"):
            IDWReconstructor(dataset, dataset.domain, k_min=10, k=2)
