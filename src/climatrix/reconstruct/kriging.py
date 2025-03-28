from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar, Literal

import numpy as np
import numpy.ma as ma
import xarray as xr

from climatrix.decorators import raise_if_not_installed
from climatrix.decorators.runtime import log_input
from climatrix.reconstruct.base import BaseReconstructor

if TYPE_CHECKING:
    from climatrix.dataset.sparse import SparseDataset

log = logging.getLogger(__name__)


class OrdinaryKrigingReconstructor(BaseReconstructor):
    """
    Reconstruct a sparse dataset using Ordinary Kriging.

    Attributes
    ----------
    dataset : SparseDataset
        The sparse dataset to reconstruct.
    query_lat : np.ndarray
        The latitude grid points.
    query_lon : np.ndarray
        The longitude grid points.
    pykrige_kwargs : dict
        Additional keyword arguments to pass to pykrige.
    backend : Literal["vectorized", "loop"] | None
        The backend to use for kriging.

    Parameters
    ----------
    dataset : SparseDataset
        The sparse dataset to reconstruct.
    lat : slice or np.ndarray, optional
        The latitude range (default is slice(-90, 90, 0.1)).
    lon : slice or np.ndarray, optional
        The longitude range (default is slice(-180, 180, 0.1)).
    backend : Literal["vectorized", "loop"] | None, optional
        The backend to use for kriging (default is None).
    pykrige_kwargs : dict, optional
        Additional keyword arguments to pass to pykrige.
    """

    _MAX_VECTORIZED_SIZE: ClassVar[int] = 5_000_000

    @log_input(log, level=logging.DEBUG)
    def __init__(
        self,
        dataset: SparseDataset,
        lat: slice | np.ndarray = slice(-90, 90, 0.1),
        lon: slice | np.ndarray = slice(-180, 180, 0.1),
        backend: Literal["vectorized", "loop"] | None = None,
        **pykrige_kwargs,
    ):
        super().__init__(dataset, lat, lon)
        if self.dataset.is_dynamic:
            raise ValueError("Kriging is not supported for dynamic datasets.")
        self.pykrige_kwargs = pykrige_kwargs
        self.backend = backend

    @raise_if_not_installed("pykrige")
    def reconstruct(self):
        """
        Perform Ordinary Kriging reconstruction of the dataset.

        Returns
        -------
        StaticDenseDataset
            The reconstructed dense dataset.

        Notes
        -----
        - The backend is chosen based on the size of the dataset.
        If the dataset is larger than the maximum size, the loop
        backend is used.
        """
        from pykrige.ok import OrdinaryKriging

        from climatrix.dataset.dense import StaticDenseDataset

        if self.backend is None:
            self.backend = (
                "vectorized"
                if self.dataset.size < self._MAX_VECTORIZED_SIZE
                else "loop"
            )
        kriging = OrdinaryKriging(
            x=self.dataset.longitude.values,
            y=self.dataset.latitude.values,
            z=self.dataset.da.values,
            **self.pykrige_kwargs,
        )
        masked_values, variance = kriging.execute(
            "grid", self.query_lon, self.query_lat, backend=self.backend
        )
        values = ma.getdata(masked_values)

        coords = {
            self.dataset.latitude_name: self.query_lat,
            self.dataset.longitude_name: self.query_lon,
        }

        return StaticDenseDataset(
            xr.DataArray(
                values,
                coords=coords,
                dims=(self.dataset.latitude_name, self.dataset.longitude_name),
                name=self.dataset.da.name,
            ),
        )
