from enum import Enum
from unittest.mock import MagicMock, patch

import pytest

from climatrix.reconstruct.idw import IDWReconstructor
from climatrix.reconstruct.type import ReconstructionType


class TestReconstructionType:

    def test_enum_idw_value(self):
        assert ReconstructionType.IDW.value == IDWReconstructor

    @pytest.mark.skipif(
        not hasattr(ReconstructionType, "OK"),
        reason="OrdinaryKrigingReconstructor is not available",
    )
    def test_enum_ok_value(self):
        from climatrix.reconstruct.kriging import OrdinaryKrigingReconstructor

        assert ReconstructionType.OK.value == OrdinaryKrigingReconstructor

    @pytest.mark.skipif(
        not hasattr(ReconstructionType, "SINET"),
        reason="SiNETReconstructor is not available",
    )
    def test_enum_sinet_value(self):
        from climatrix.reconstruct.sinet.sinet import SiNETReconstructor

        assert ReconstructionType.SINET.value == SiNETReconstructor

    @pytest.mark.skipif(
        not hasattr(ReconstructionType, "SIREN"),
        reason="SIRENReconstructor is not available",
    )
    def test_enum_siren_value(self):
        from climatrix.reconstruct.siren.siren import SIRENReconstructor

        assert ReconstructionType.SIREN.value == SIRENReconstructor

    def test_missing_method(self):
        with pytest.raises(
            ValueError,
            match="'invalid_method' is not a valid ReconstructionType",
        ):
            ReconstructionType("invalid_method")

    def test_get_method_with_enum_instance(self):
        rt = ReconstructionType.IDW
        result = ReconstructionType.get(rt)
        assert result is rt
        assert result == ReconstructionType.IDW

    def test_get_method_with_string_case_insensitive(self):
        result = ReconstructionType.get("IDW")
        assert result == ReconstructionType.IDW

        result = ReconstructionType.get("idw")
        assert result == ReconstructionType.IDW

        result = ReconstructionType.get("IdW")
        assert result == ReconstructionType.IDW

    def test_get_method_with_invalid_string(self):
        with pytest.raises(
            ValueError, match="Unknown reconstruction type: invalid_type"
        ):
            ReconstructionType.get("invalid_type")

    def test_get_method_with_invalid_type(self):
        with pytest.raises(TypeError, match="Invalid reconstruction type"):
            ReconstructionType.get(123)

        with pytest.raises(TypeError, match="Invalid reconstruction type"):
            ReconstructionType.get(None)

    def test_get_method_with_all_valid_keys(self):
        for rt in ReconstructionType:
            assert ReconstructionType.get(rt) == rt

            assert ReconstructionType.get(rt.name) == rt

    def test_enum_lifecycle(self):
        rt = ReconstructionType.get("idw")
        assert rt == ReconstructionType.IDW

        reconstructor_class = rt.value
        assert reconstructor_class == IDWReconstructor

        with patch.object(IDWReconstructor, "__init__", return_value=None):
            instance = reconstructor_class()
            assert isinstance(instance, IDWReconstructor)

    def test_list_method(self):
        available_types = ReconstructionType.list()
        assert isinstance(available_types, list)
        "IDW" in available_types
