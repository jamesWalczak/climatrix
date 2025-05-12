from __future__ import annotations

import logging
from typing import ClassVar, Literal

import numpy.ma as ma

from climatrix.dataset.base import BaseClimatrixDataset
from climatrix.dataset.domain import Domain
from climatrix.decorators import raise_if_not_installed
from climatrix.decorators.runtime import log_input
from climatrix.reconstruct.base import BaseReconstructor

log = logging.getLogger(__name__)


class OrdinaryKrigingReconstructor(BaseReconstructor):
    """
    Reconstruct a sparse dataset using Ordinary Kriging.

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

    Parameters
    ----------
    dataset : SparseDataset
        The sparse dataset to reconstruct.
    target_domain : Domain
        The target domain for reconstruction.
    backend : Literal["vectorized", "loop"] | None, optional
        The backend to use for kriging (default is None).
    pykrige_kwargs : dict, optional
        Additional keyword arguments to pass to pykrige.
    """

    _MAX_VECTORIZED_SIZE: ClassVar[int] = 500_000

    @log_input(log, level=logging.DEBUG)
    def __init__(
        self,
        dataset: BaseClimatrixDataset,
        target_domain: Domain,
        backend: Literal["vectorized", "loop"] | None = None,
        **pykrige_kwargs,
    ):
        super().__init__(dataset, target_domain)
        if self.dataset.domain.is_dynamic:
            log.warning(
                "Calling ordinary kriging on dynamic datasets, which "
                "are not yet supported."
            )
            raise NotImplementedError(
                "Kriging is not supported for " "dynamic datasets."
            )
        if not self.dataset.domain.is_sparse:
            log.warning(
                "Calling ordinary kriging on dense datasets, whcih "
                "are not yet supported."
            )
            raise NotImplementedError(
                "Cannot carry out kriging for " "dense dataset."
            )
        self.pykrige_kwargs = pykrige_kwargs
        self.backend = backend

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
            log.info("Choosing backend based on dataset size...")
            self.backend = (
                "vectorized"
                if (
                    len(self.target_domain.latitude)
                    * len(self.target_domain.longitude)
                )
                < self._MAX_VECTORIZED_SIZE
                else "loop"
            )
            log.debug("Using backend: %s", self.backend)
        kriging = OrdinaryKriging(
            x=self.dataset.domain.longitude,
            y=self.dataset.domain.latitude,
            z=self.dataset.da.values.astype(float).squeeze(),
            **self.pykrige_kwargs,
        )
        log.debug("Performing Ordinary Kriging reconstruction...")
        recon_type = "points" if self.target_domain.is_sparse else "grid"
        log.debug("Reconstruction type: %s", recon_type)
        masked_values, _ = kriging.execute(
            recon_type,
            self.target_domain.longitude,
            self.target_domain.latitude,
            backend=self.backend,
        )
        values = ma.getdata(masked_values)

        log.debug("Reconstruction completed.")
        return BaseClimatrixDataset(
            self.target_domain.to_xarray(values, self.dataset.da.name)
        )
