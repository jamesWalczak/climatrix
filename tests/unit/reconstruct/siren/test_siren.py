import pytest

try:
    import torch

    from climatrix.reconstruct.siren.siren import SIRENReconstructor
except ImportError:
    pytest.skip(
        "SIRENReconstructor is not available. "
        "Please install the required dependencies.",
        allow_module_level=True,
    )
from tests.unit.utils import skip_on_error

from ..test_base_interface import TestBaseReconstructor, parametrize_all


class TestSIRENReconstructor(TestBaseReconstructor):
    __test__ = True

    @pytest.fixture
    def reconstructor_class(self):
        return SIRENReconstructor

    @parametrize_all()
    @skip_on_error(NotImplementedError)
    def test_falls_back_to_cpu_when_cuda_unavailable(
        self, dataset, monkeypatch
    ):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        reconstructor = SIRENReconstructor(
            dataset, dataset.domain, device="cuda"
        )

        assert reconstructor.device == torch.device("cpu")
