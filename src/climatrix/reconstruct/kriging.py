from __future__ import annotations

import logging
from typing import ClassVar, Literal

import numpy as np
import numpy.ma as ma

from climatrix.dataset.base import AxisType, BaseClimatrixDataset
from climatrix.dataset.domain import Domain
from climatrix.decorators import raise_if_not_installed
from climatrix.decorators.runtime import log_input
from climatrix.optim.hyperparameter import Hyperparameter
from climatrix.reconstruct.base import BaseReconstructor

log = logging.getLogger(__name__)


class OrdinaryKrigingReconstructor(BaseReconstructor):
    """
    Reconstruct a sparse dataset using Ordinary Kriging.

    This class performs spatial interpolation using ordinary kriging,
    a geostatistical method that provides optimal linear unbiased estimation
    by modeling spatial correlation through variograms.

    Parameters
    ----------
    dataset : SparseDataset
        The sparse dataset to reconstruct.
    target_domain : Domain
        The target domain for reconstruction.
    backend : Literal["vectorized", "loop"] | None, optional
        The backend to use for kriging (default is None).
    nlags : int | None, optional
        Number of lags for variogram computation (default is 6).
        Type: int, bounds: (0, ...), default: 6
    anisotropy_scaling : float | None, optional
        Anisotropy scaling factor (default is 1e-6).
        Type: float, bounds: <unbounded>, default: 1e-6
    coordinates_type : str | None, optional
        Type of coordinate system (default is "euclidean").
        Type: str, values: ["euclidean", "geographic"], default: "euclidean"
    variogram_model : str | None, optional
        Variogram model to use (default is "linear").
        Type: str, values: ["linear", "power", "gaussian", "spherical", "exponential"], default: "linear"
    pseudo_inv : bool, optional
        Whether to use pseudo-inverse for matrix operations (default is False).

    Attributes
    ----------
    dataset : SparseDataset
        The sparse dataset to reconstruct.
    domain : Domain
        The target domain for reconstruction.
    pykrige_kwargs : dict
        Additional keyword arguments to pass to pykrige.
    backend : Literal["vectorized", "loop"] | None
        The backend to use for kriging.
    _MAX_VECTORIZED_SIZE : ClassVar[int]
        The maximum size for vectorized kriging.
        If the dataset is larger than this size, loop kriging
        will be used (if `backend` was not specified)

    Notes
    -----
    Hyperparameters for optimization:
        - nlags: int in (4, 20), default=6
        - anisotropy_scaling: float in (1e-6, 5.0), default=1e-6
        - coordinates_type: str in ["euclidean", "geographic"], default="euclidean"
        - variogram_model: str in ["linear", "power", "gaussian", "spherical", "exponential"], default="linear"
    """

    NAME: ClassVar[str] = "ok"
    _MAX_VECTORIZED_SIZE: ClassVar[int] = 500_000

    nlags = Hyperparameter[int](bounds=(4, 20), default=6)
    anisotropy_scaling = Hyperparameter[float](
        bounds=(1e-6, 5.0), default=1e-6
    )
    coordinates_type = Hyperparameter[str](
        values=["euclidean", "geographic"], default="euclidean"
    )
    variogram_model = Hyperparameter[str](
        values=["linear", "power", "gaussian", "spherical", "exponential"],
        default="linear",
    )

    pseudo_inv: bool
    backend: Literal["vectorized", "loop"] | None

    @log_input(log, level=logging.DEBUG)
    def __init__(
        self,
        dataset: BaseClimatrixDataset,
        target_domain: Domain,
        backend: Literal["vectorized", "loop"] | None = None,
        nlags: int | None = None,
        anisotropy_scaling: float | None = None,
        coordinates_type: str | None = None,
        variogram_model: str | None = None,
        pseudo_inv: bool = False,
    ):
        super().__init__(dataset, target_domain)
        for axis in dataset.domain.all_axes_types:
            if not dataset.domain.get_axis(axis).is_dimension:
                continue
            if axis in [AxisType.LATITUDE, AxisType.LONGITUDE, AxisType.POINT]:
                continue
            elif dataset.domain.get_size(axis) == 1:
                continue
            log.error(
                "Currently, IDWReconstructor only supports datasets with "
                "latitude and longitude dimensions, but got '%s'",
                axis,
            )
            raise NotImplementedError(
                "Currently, IDWReconstructor only supports datasets with "
                f"latitude and longitude dimensions, but got '{axis}'"
            )
        if not self.dataset.domain.is_sparse:
            log.warning(
                "Calling ordinary kriging on dense datasets, whcih "
                "are not yet supported."
            )
            raise NotImplementedError(
                "Cannot carry out kriging for " "dense dataset."
            )
        if coordinates_type == "geographic":
            log.info(
                "Using geographic coordinates for kriging "
                "reconstruction. Moving to positive-only "
                "longitude convention."
            )
            self.dataset = self.dataset.to_positive_longitude()
        self.backend = backend
        self.nlags = nlags or self.nlags
        self.anisotropy_scaling = anisotropy_scaling or self.anisotropy_scaling
        self.coordinates_type = coordinates_type or self.coordinates_type
        self.variogram_model = variogram_model or self.variogram_model
        self.pseudo_inv = pseudo_inv

    def _normalize_latitude(self, lat: np.ndarray) -> np.ndarray:
        _lat_max = np.nanmax(lat)
        _lat_min = np.nanmin(lat)
        scaled = (lat - _lat_min) / (_lat_max - _lat_min)
        scaled -= 0.5
        scaled *= 2
        return _lat_min, _lat_max, scaled

    def _normalize_longitude(self, lon: np.ndarray) -> np.ndarray:
        _lon_max = np.nanmax(lon)
        _lon_min = np.nanmin(lon)
        scaled = (lon - _lon_min) / (_lon_max - _lon_min)
        scaled -= 0.5
        scaled *= 2
        return _lon_min, _lon_max, scaled

    def _standarize_values(self, values: np.ndarray) -> np.ndarray:
        _values_mean = np.nanmean(values)
        _values_std = np.nanstd(values)
        if _values_std == 0:
            log.warning(
                "Standard deviation of values is zero, "
                "normalizing to zero mean and unit variance."
            )
            return values - _values_mean
        scaled = (values - _values_mean) / _values_std
        return _values_mean, _values_std, scaled

    @raise_if_not_installed("pykrige")
    def reconstruct(self) -> BaseClimatrixDataset:
        """
        Perform Ordinary Kriging reconstruction of the dataset.

        Returns
        -------
        BaseClimatrixDataset
            The dataset reconstructed on the target domain.

        Notes
        -----
        - The backend is chosen based on the size of the dataset.
        If the dataset is larger than the maximum size, the loop
        backend is used.
        """
        from pykrige.ok import OrdinaryKriging

        if self.backend is None:
            log.debug("Choosing backend based on dataset size...")
            self.backend = (
                "vectorized"
                if (
                    len(self.target_domain.latitude.values)
                    * len(self.target_domain.longitude.values)
                )
                < self._MAX_VECTORIZED_SIZE
                else "loop"
            )
            log.debug("Using backend: %s", self.backend)

        log.debug("Normalizing latitude and longitude values to [-1, 1]...")
        *_, lat = self._normalize_latitude(self.dataset.domain.latitude.values)
        *_, lon = self._normalize_longitude(
            self.dataset.domain.longitude.values
        )
        log.debug("Standardizing values to mean=0, std=1...")
        v_mean, v_std, values = self._standarize_values(
            self.dataset.da.values.astype(float).squeeze()
        )
        kriging = OrdinaryKriging(
            x=lon,
            y=lat,
            z=values,
            nlags=self.nlags,
            anisotropy_scaling=self.anisotropy_scaling,
            coordinates_type=self.coordinates_type,
            variogram_model=self.variogram_model,
            pseudo_inv=self.pseudo_inv,
        )
        log.debug("Performing Ordinary Kriging reconstruction...")
        recon_type = "points" if self.target_domain.is_sparse else "grid"
        log.debug("Reconstruction type: %s", recon_type)

        log.debug("Normalizing target domain latitude and longitude values...")
        *_, target_lat = self._normalize_latitude(
            self.target_domain.latitude.values
        )
        *_, target_lon = self._normalize_longitude(
            self.target_domain.longitude.values
        )
        masked_values, _ = kriging.execute(
            recon_type,
            target_lon,
            target_lat,
            backend=self.backend,
        )
        values = ma.getdata(masked_values)

        log.debug("Denormalizing values to original scale...")
        values = values * v_std + v_mean

        log.debug("Reconstruction completed.")
        return BaseClimatrixDataset(
            self.target_domain.to_xarray(values, self.dataset.da.name)
        )
