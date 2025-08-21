import logging
import os
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch
import torch.nn.functional as F

from climatrix.dataset.base import BaseClimatrixDataset
from climatrix.dataset.domain import Domain
from climatrix.decorators.runtime import log_input, raise_if_not_installed
from climatrix.optim.hyperparameter import Hyperparameter
from climatrix.reconstruct.mmgn.dataset import MMGNDatasetGenerator
from climatrix.reconstruct.mmgn.model import (
    MMGNet,
    _FilterType,
    _LatentInitType,
)
from climatrix.reconstruct.nn.base_nn import BaseNNReconstructor

log = logging.getLogger(__name__)


class MMGNReconstructor(BaseNNReconstructor):
    """MMGN Reconstructor class."""

    NAME: ClassVar[str] = "mmgn"

    # Hyperparameters definitions
    weight_decay = Hyperparameter[float](bounds=(0.0, 1.0), default=1e-5)
    hidden_dim = Hyperparameter[int](
        default=256, values=[32, 64, 128, 256, 512, 1024]
    )
    latent_dim = Hyperparameter[int](bounds=(1, None), default=128)
    n_layers = Hyperparameter[int](bounds=(1, 50), default=5)
    input_scale = Hyperparameter[int](bounds=(1, None), default=256)
    alpha = Hyperparameter[float](bounds=(0, 1), default=1.0)
    filter_type = Hyperparameter[str](
        default="gabor", values=_FilterType.choices()
    )
    latent_init = Hyperparameter[str](
        default="zeros", values=_LatentInitType.choices()
    )

    gamma: float = 0.99

    @log_input(log, level=logging.DEBUG)
    def __init__(
        self,
        dataset: BaseClimatrixDataset,
        target_domain: Domain,
        *,
        checkpoint: str | os.PathLike | Path | None = None,
        device: str = "cuda",
        lr: float = 5e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 64,
        num_epochs: int = 1000,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        n_layers: int = 5,
        input_scale: int = 256,
        alpha: float = 1.0,
        validation: float | BaseClimatrixDataset | None = None,
        overwrite_checkpoint: bool = False,
        filter_type: str = "gabor",
        latent_init: str = "zeros",
        num_workers: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(
            dataset,
            target_domain,
            lr=lr,
            num_epochs=num_epochs,
            batch_size=batch_size,
            checkpoint=checkpoint,
            overwrite_checkpoint=overwrite_checkpoint,
            num_workers=num_workers,
            device=device,
        )
        if dataset.domain.is_dynamic:
            log.error("MMGN does not support dynamic domains.")
            raise ValueError("MMGN does not support dynamic domains.")
        self.datasets = self._configure_dataset_generator(
            train_coords=dataset.domain.get_all_spatial_points(),
            train_field=dataset.da.values,
            target_coords=target_domain.get_all_spatial_points(),
            val_portion=validation if isinstance(validation, float) else None,
            val_coordinates=(
                (validation.domain.get_all_spatial_points())
                if isinstance(validation, BaseClimatrixDataset)
                else None
            ),
            val_field=(
                (validation.da.values)
                if isinstance(validation, BaseClimatrixDataset)
                else None
            ),
        )

        self.weight_decay = weight_decay

        self.input_dim = 2
        self.out_dim = 1
        self.n_data = 1  # NOTE: a single timestamp
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.input_scale = input_scale
        self.alpha = alpha
        self.filter_type = filter_type
        self.latent_init = latent_init

    @staticmethod
    def _configure_dataset_generator(
        train_coords: np.ndarray,
        train_field: np.ndarray,
        target_coords: np.ndarray,
        val_portion: float | None = None,
        val_coordinates: np.ndarray | None = None,
        val_field: np.ndarray | None = None,
    ) -> MMGNDatasetGenerator:
        """
        Configure the MMGN dataset generator.
        """
        log.debug("Configuring MMGN dataset generator...")
        if val_portion is not None and (
            val_coordinates is not None or val_field is not None
        ):
            log.error(
                "Cannot use both `val_portion` and `val_coordinates`/`val_field`."
            )
            raise ValueError(
                "Cannot use both `val_portion` and `val_coordinates`/`val_field`."
            )
        kwargs = {
            "spatial_points": train_coords,
            "field": train_field,
        }
        if val_portion is not None:
            if not (0 <= val_portion < 1):
                log.error("Validation portion must be in the range (0, 1).")
                raise ValueError(
                    "Validation portion must be in the range (0, 1)."
                )
            log.debug("Using validation portion: %0.2f", val_portion)
            kwargs["val_portion"] = val_portion
        elif val_coordinates is not None and val_field is not None:
            log.debug("Using validation coordinates and field for validation.")
            if val_coordinates.shape[0] != val_field.shape[0]:
                log.error(
                    "Validation coordinates and field must have the same number of points."
                )
                raise ValueError(
                    "Validation coordinates and field must have the same number of points."
                )
            if val_coordinates.shape[1] != train_coords.shape[1]:
                log.error(
                    "Validation coordinates must have the same number of dimensions as training coordinates."
                )
                raise ValueError(
                    "Validation coordinates must have the same number of dimensions as training coordinates."
                )
            kwargs["validation_coordinates"] = val_coordinates
            kwargs["validation_field"] = val_field
        if target_coords is not None:
            if (
                target_coords.ndim != 2
                or target_coords.shape[1] != train_coords.shape[1]
            ):
                log.error(
                    "Target coordinates must be a 2D array with shape (n_samples, 2)."
                )
                raise ValueError(
                    "Target coordinates must be a 2D array with shape (n_samples, 2)."
                )
            kwargs["target_coordinates"] = target_coords

        return MMGNDatasetGenerator(**kwargs)

    def configure_optimizer(
        self, mmgn_module: torch.nn.Module
    ) -> torch.optim.Optimizer:
        log.info(
            "Configuring Adam optimizer with learning rate: %0.6f",
            self.lr,
        )
        return torch.optim.AdamW(
            lr=self.lr,
            params=mmgn_module.parameters(),
            weight_decay=self.weight_decay,
        )

    def configure_epoch_schedulers(self, optimizer):
        return [
            torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.gamma)
        ]

    def init_model(self) -> torch.nn.Module:
        log.info("Initializing MMGN model...")
        return MMGNet(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            out_dim=self.out_dim,
            n_layers=self.n_layers,
            input_scale=self.input_scale,
            alpha=self.alpha,
            filter_type=_FilterType.get(self.filter_type),
            latent_init=_LatentInitType.get(self.latent_init),
        ).to(self.device)

    def compute_loss(
        self, xy: torch.Tensor, pred_z: torch.Tensor, true_z: torch.Tensor
    ) -> torch.Tensor:
        return F.l1_loss(pred_z, true_z)

    @raise_if_not_installed("torch")
    def reconstruct(self) -> BaseClimatrixDataset:
        return super().reconstruct()
