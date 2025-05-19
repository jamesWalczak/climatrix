from datetime import datetime
from functools import partial

import numpy as np
import pytest
import xarray as xr

from climatrix import BaseClimatrixDataset
from tests.unit.utils import skip_on_error

parametrize_all = partial(
    pytest.mark.parametrize,
    "dataset",
    [
        "sparse_static",
        "dense_static",
        "sparse_dynamic",
        "dense_dynamic",
    ],
    indirect=True,
)


class TestBaseReconstructor:
    __test__ = False

    def create_sparse_dynamic_dataset(self):
        return BaseClimatrixDataset(
            xr.DataArray(
                data=np.random.rand(5, 3),
                dims=("point", "time"),
                coords={
                    "point": np.arange(5),
                    "time": np.array(
                        [
                            datetime(2000, 1, 1),
                            datetime(2000, 1, 2),
                            datetime(2000, 1, 3),
                        ],
                        dtype="datetime64",
                    ),
                    "latitude": (("point",), np.array([-90, -45, 0, 45, 90])),
                    "longitude": (
                        ("point",),
                        np.array([-180, -90, 0, 90, 180]),
                    ),
                },
            )
        )

    def create_dense_dynamic_dataset(self):
        return BaseClimatrixDataset(
            xr.DataArray(
                data=np.random.rand(3, 5, 5),
                dims=("time", "latitude", "longitude"),
                coords={
                    "time": np.array(
                        [
                            datetime(2000, 1, 1),
                            datetime(2000, 1, 2),
                            datetime(2000, 1, 3),
                        ],
                        dtype="datetime64",
                    ),
                    "latitude": (
                        ("latitude",),
                        np.array([-90, -45, 0, 45, 90]),
                    ),
                    "longitude": (
                        ("longitude",),
                        np.array([-180, -90, 0, 90, 180]),
                    ),
                },
            )
        )

    def create_sparse_static_dataset(self):
        return BaseClimatrixDataset(
            xr.DataArray(
                data=np.random.rand(5, 1),
                dims=("point", "time"),
                coords={
                    "point": np.arange(5),
                    "time": np.array(
                        [
                            datetime(2000, 1, 1),
                        ],
                        dtype="datetime64",
                    ),
                    "latitude": (("point",), np.array([-90, -45, 0, 45, 90])),
                    "longitude": (
                        ("point",),
                        np.array([-180, -90, 0, 90, 180]),
                    ),
                },
            )
        )

    def create_dense_static_dataset(self):
        return BaseClimatrixDataset(
            xr.DataArray(
                data=np.random.rand(1, 5, 5),
                dims=("time", "latitude", "longitude"),
                coords={
                    "time": np.array(
                        [
                            datetime(2000, 1, 1),
                        ],
                        dtype="datetime64",
                    ),
                    "latitude": (
                        ("latitude",),
                        np.array([-90, -45, 0, 45, 90]),
                    ),
                    "longitude": (
                        ("longitude",),
                        np.array([-180, -90, 0, 90, 180]),
                    ),
                },
            )
        )

    @pytest.fixture
    def dataset(self, request):
        if request.param == "sparse_static":
            yield self.create_sparse_static_dataset()
        elif request.param == "dense_static":
            yield self.create_dense_static_dataset()
        elif request.param == "sparse_dynamic":
            yield self.create_sparse_dynamic_dataset()
        elif request.param == "dense_dynamic":
            yield self.create_dense_dynamic_dataset()

    @pytest.fixture
    def reconstructor_class(self):
        pass

    @pytest.fixture
    def reconstructor(self, reconstructor_class, dataset):
        try:
            return reconstructor_class(dataset, dataset.domain)
        except NotImplementedError as e:
            pytest.skip(f"Unsupported configuration: {e}")

    @pytest.mark.parametrize(
        "dataset",
        [
            "sparse_static",
            "dense_static",
            "sparse_dynamic",
            "dense_dynamic",
        ],
        indirect=True,
    )
    def test_init_with_required_inputs_dense_dataset(
        self, reconstructor_class, dataset
    ):
        try:
            reconstructor = reconstructor_class(dataset, dataset.domain)
        except NotImplementedError as e:
            pytest.skip(f"Unsupported configuration: {e}")
        assert reconstructor.dataset is dataset
        assert reconstructor.target_domain is dataset.domain
        assert (
            reconstructor.target_domain.is_sparse == dataset.domain.is_sparse
        )

    @pytest.mark.parametrize(
        "dataset",
        [
            "sparse_static",
            "dense_static",
            "sparse_dynamic",
            "dense_dynamic",
        ],
        indirect=True,
    )
    def test_init_with_invalid_dataset_type(
        self, reconstructor_class, dataset
    ):
        with pytest.raises(
            TypeError, match="dataset must be a BaseClimatrixDataset object"
        ):
            reconstructor_class("not_a_dataset", dataset.domain)

    @pytest.mark.parametrize(
        "dataset",
        [
            "sparse_static",
            "dense_static",
            "sparse_dynamic",
            "dense_dynamic",
        ],
        indirect=True,
    )
    def test_init_with_invalid_domain(self, reconstructor_class, dataset):
        with pytest.raises(TypeError, match="domain must be a Domain object"):
            reconstructor_class(dataset, "not_a_domain")

    @pytest.mark.parametrize(
        "dataset",
        [
            "sparse_static",
            "dense_static",
            "sparse_dynamic",
            "dense_dynamic",
        ],
        indirect=True,
    )
    def test_reconstruct_method_exists(self, reconstructor):
        assert hasattr(reconstructor, "reconstruct")
        assert callable(reconstructor.reconstruct)

    @pytest.mark.parametrize(
        "dataset",
        [
            "sparse_static",
            "dense_static",
            "sparse_dynamic",
            "dense_dynamic",
        ],
        indirect=True,
    )
    def test_reconstruction_returns_climatrix_dataset(self, reconstructor):
        result = reconstructor.reconstruct()
        assert isinstance(result, BaseClimatrixDataset)

    @skip_on_error(NotImplementedError)
    def test_reconstruct_static_sparse_from_static_dense(
        self, reconstructor_class
    ):
        dense_dataset = self.create_dense_static_dataset()
        sparse_domain = self.create_sparse_static_dataset().domain
        reconstructor = reconstructor_class(dense_dataset, sparse_domain)
        reconstructed_dataset = reconstructor.reconstruct()
        assert isinstance(reconstructed_dataset, BaseClimatrixDataset)
        assert reconstructed_dataset.domain.is_sparse

    @skip_on_error(NotImplementedError)
    def test_reconstruct_static_dense_from_static_sparse(
        self, reconstructor_class
    ):
        sparse_dataset = self.create_sparse_static_dataset()
        dense_domain = self.create_dense_static_dataset().domain
        reconstructor = reconstructor_class(sparse_dataset, dense_domain)
        reconstructed_dataset = reconstructor.reconstruct()
        assert isinstance(reconstructed_dataset, BaseClimatrixDataset)
        assert not reconstructed_dataset.domain.is_sparse

    @skip_on_error(NotImplementedError)
    def test_reconstruct_static_dense_from_static_dense(
        self, reconstructor_class
    ):
        dense_dataset = self.create_dense_static_dataset()
        reconstructor = reconstructor_class(
            dense_dataset, dense_dataset.domain
        )
        reconstructed_dataset = reconstructor.reconstruct()
        assert isinstance(reconstructed_dataset, BaseClimatrixDataset)
        assert not reconstructed_dataset.domain.is_sparse

    @skip_on_error(NotImplementedError)
    def test_reconstruct_static_sparse_from_static_sparse(
        self, reconstructor_class
    ):
        sparse_dataset = self.create_sparse_static_dataset()
        reconstructor = reconstructor_class(
            sparse_dataset, sparse_dataset.domain
        )
        reconstructed_dataset = reconstructor.reconstruct()
        assert isinstance(reconstructed_dataset, BaseClimatrixDataset)
        assert reconstructed_dataset.domain.is_sparse
