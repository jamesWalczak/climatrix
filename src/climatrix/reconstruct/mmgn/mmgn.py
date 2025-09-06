import logging
import os
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LRScheduler

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
    dataset_generator_type = MMGNDatasetGenerator

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
            weight_decay=weight_decay,
            num_epochs=num_epochs,
            batch_size=batch_size,
            checkpoint=checkpoint,
            overwrite_checkpoint=overwrite_checkpoint,
            num_workers=num_workers,
            device=device,
            validation=validation,
        )
        if dataset.domain.is_dynamic:
            log.error("MMGN does not support dynamic domains.")
            raise ValueError("MMGN does not support dynamic domains.")

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

    def configure_optimizer(
        self, nn_model: torch.nn.Module
    ) -> torch.optim.Optimizer:
        log.info(
            "Configuring Adam optimizer with learning rate: %0.6f",
            self.lr,
        )
        return torch.optim.AdamW(
            lr=self.lr,
            params=nn_model.parameters(),
            weight_decay=self.weight_decay,
        )

    def configure_epoch_schedulers(self, optimizer) -> list[LRScheduler]:
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
