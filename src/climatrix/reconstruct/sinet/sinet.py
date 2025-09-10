from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import torch

from climatrix.decorators.runtime import log_input, raise_if_not_installed
from climatrix.optim.hyperparameter import Hyperparameter
from climatrix.reconstruct.nn.base_nn import BaseNNReconstructor
from climatrix.reconstruct.sinet.dataset import (
    SiNETDatasetGenerator,
)
from climatrix.reconstruct.sinet.losses import LossEntity, compute_sdf_losses
from climatrix.reconstruct.sinet.model import SiNET

if TYPE_CHECKING:
    from climatrix import Domain
    from climatrix.dataset.base import BaseClimatrixDataset

log = logging.getLogger(__name__)


class SiNETReconstructor(BaseNNReconstructor):
    """
    Spatial Interpolation Network (SiNET) Reconstructor.

    SiNET is a neural network-based method for spatial interpolation that
    uses implicit neural representations to reconstruct continuous fields
    from sparse observations.

    Parameters
    ----------
    dataset : BaseClimatrixDataset
        Source dataset to reconstruct from.
    target_domain : Domain
        Target domain to reconstruct onto.
    layers : int, optional
        Number of hidden layers in the network (default is 2).
    hidden_dim : int, optional
        Number of neurons in each hidden layer (default is 64).
    sorting_group_size : int, optional
        Size of sorting groups for data processing (default is 16).
    scale : float, optional
        Scaling factor for coordinates (default is 1.5).
    lr : float, optional
        Learning rate for optimization (default is 3e-4).
        Type: float, bounds: <unbounded>, default: 1e-3
    batch_size : int, optional
        Batch size for training (default is 512).
        Type: int, bounds: <unbounded>, default: 128
    num_epochs : int, optional
        Number of training epochs (default is 5000).
        Type: int, bounds: <unbounded>, default: 5_000
    num_workers : int, optional
        Number of worker processes for data loading (default is 0).
    device : str, optional
        Device to run computation on (default is "cuda").
    gradient_clipping_value : float | None, optional
        Value for gradient clipping (default is None).
        Type: float, bounds: <unbounded>, default: 1.0
    checkpoint : str | os.PathLike | Path | None, optional
        Path to model checkpoint (default is None).
    mse_loss_weight : float, optional
        Weight for MSE loss component (default is 3e3).
        Type: float, bounds: <unbounded>, default: 1e2
    eikonal_loss_weight : float, optional
        Weight for Eikonal loss component (default is 5e1).
        Type: float, bounds: <unbounded>, default: 1e1
    laplace_loss_weight : float, optional
        Weight for Laplace loss component (default is 1e2).
        Type: float, bounds: <unbounded>, default: 1e2
    validation : float | BaseClimatrixDataset, optional
        Validation data or portion for training (default is 0.0).
    patience : int | None, optional
        Early stopping patience (default is None).
    overwrite_checkpoint : bool, optional
        Whether to overwrite existing checkpoints (default is False).

    Raises
    ------
    ValueError
        If SiNET is used with dynamic datasets or if CUDA is not available
        when requested.

    Notes
    -----
    Hyperparameters for optimization:
        - lr: float in (1e-5, 1e-2), default=1e-3
        - batch_size: int in (64, 1024), default=128
        - num_epochs: int in (10, 10_000), default=5_000
        - mse_loss_weight: float in (1e1, 1e4), default=1e2
        - eikonal_loss_weight: float in (1e0, 1e3), default=1e1
        - laplace_loss_weight: float in (1e1, 1e3), default=1e2
    """

    NAME: ClassVar[str] = "sinet"
    dataset_generator_type = SiNETDatasetGenerator

    mse_loss_weight = Hyperparameter[float](default=1e2)
    eikonal_loss_weight = Hyperparameter[float](default=1e1)
    laplace_loss_weight = Hyperparameter[float](default=1e2)
    scale = Hyperparameter[float](bounds=(0.01, 10.0), default=1.5)
    hidden_dim = Hyperparameter[int](default=64, values=[16, 32, 64, 128, 256])
    _was_early_stopped: ClassVar[bool] = False

    @log_input(log, level=logging.DEBUG)
    def __init__(
        self,
        dataset: BaseClimatrixDataset,
        target_domain: Domain,
        *,
        lr: float = 3e-4,
        weight_decay: float = 0.0,
        batch_size: int = 512,
        num_epochs: int = 5_000,
        num_workers: int = 0,
        device: str = "cuda",
        gradient_clipping_value: float | None = None,
        checkpoint: str | os.PathLike | Path | None = None,
        overwrite_checkpoint: bool = False,
        patience: int | None = None,
        # Model specific parameters
        layers: int = 2,
        hidden_dim: int = 64,
        sorting_group_size: int = 16,
        scale: float = 1.5,
        mse_loss_weight: float = 1.0,
        eikonal_loss_weight: float = 0,
        laplace_loss_weight: float = 0,
        validation: float | BaseClimatrixDataset | None = None,
    ) -> None:
        self._custom_dataset_generator_kwargs = {
            "use_elevation": False,
        }
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
            gradient_clipping_value=gradient_clipping_value,
            device=device,
            patience=patience,
            validation=validation,
        )
        if dataset.domain.is_dynamic:
            log.error("SiNET is not yet supported for dynamic datasets.")
            raise ValueError(
                "SiNET is not yet supported for dynamic datasets."
            )
        self.layers = layers
        self.hidden_dim = hidden_dim
        self.sorting_group_size = sorting_group_size
        self.scale = scale

        self.mse_loss_weight = mse_loss_weight
        self.eikonal_loss_weight = eikonal_loss_weight
        self.laplace_loss_weight = laplace_loss_weight

    def configure_optimizer(
        self, nn_model: torch.nn.Module
    ) -> torch.optim.Optimizer:
        log.info(
            "Configuring Adam optimizer with learning rate: %0.6f",
            self.lr,
        )
        return torch.optim.Adam(
            lr=self.lr,
            params=nn_model.parameters(),
            weight_decay=self.weight_decay,
        )

    def init_model(self) -> torch.nn.Module:
        log.info("Initializing SiNET model...")
        return SiNET(
            in_features=self.datasets.n_features,
            out_features=1,
            layers=self.layers,
            hidden_dim=self.hidden_dim,
            sorting_group_size=self.sorting_group_size,
            scale=self.scale,
            bias=True,
        ).to(self.device)

    @log_input(log, level=logging.DEBUG)
    def compute_loss(
        self,
        xy: torch.Tensor,
        pred_z: torch.Tensor,
        true_z: torch.Tensor,
    ) -> torch.Tensor:
        loss_component: LossEntity = compute_sdf_losses(
            xy,
            pred_z * self.datasets.field_transformer.data_range_[0]
            + self.datasets.field_transformer.data_min_[0],
            true_z * self.datasets.field_transformer.data_range_[0]
            + self.datasets.field_transformer.data_min_[0],
        )

        return (
            loss_component.mse * self.mse_loss_weight
            + loss_component.eikonal * self.eikonal_loss_weight
            + loss_component.laplace * self.laplace_loss_weight
        )

    @raise_if_not_installed("torch")
    def reconstruct(self) -> BaseClimatrixDataset:
        """Reconstruct the sparse dataset using INR."""
        return super().reconstruct()
