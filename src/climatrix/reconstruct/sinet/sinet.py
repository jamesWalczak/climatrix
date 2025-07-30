from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from climatrix import BaseClimatrixDataset, Domain
from climatrix.decorators.runtime import log_input, raise_if_not_installed
from climatrix.reconstruct.base import BaseReconstructor
from climatrix.reconstruct.sinet.callbacks import EarlyStopping
from climatrix.reconstruct.sinet.dataset import (
    SiNETDatasetGenerator,
)
from climatrix.reconstruct.sinet.losses import LossEntity, compute_sdf_losses
from climatrix.reconstruct.sinet.model import SiNET
from climatrix.utils.hyperparameter import Hyperparameter

log = logging.getLogger(__name__)


class SiNETReconstructor(BaseReconstructor):
    NAME: ClassVar[str] = "sinet"

    lr = Hyperparameter(float, bounds=(1e-5, 1e-2), default=1e-3)
    batch_size = Hyperparameter(int, bounds=(64, 1024), default=128)
    num_epochs = Hyperparameter(int, bounds=(10, 10_000), default=5_000)
    gradient_clipping_value = Hyperparameter(
        float, bounds=(0.1, 10.0), default=1.0
    )
    mse_loss_weight = Hyperparameter(float, bounds=(1e1, 1e4), default=1e2)
    eikonal_loss_weight = Hyperparameter(float, bounds=(1e0, 1e3), default=1e1)
    laplace_loss_weight = Hyperparameter(float, bounds=(1e1, 1e3), default=1e2)
    _was_early_stopped: ClassVar[bool] = False

    @log_input(log, level=logging.DEBUG)
    def __init__(
        self,
        dataset: BaseClimatrixDataset,
        target_domain: Domain,
        *,
        layers: int = 2,
        hidden_dim: int = 64,
        sorting_group_size: int = 16,
        scale: float = 1.5,
        lr: float = 3e-4,
        batch_size: int = 512,
        num_epochs: int = 5_000,
        num_workers: int = 0,
        device: str = "cuda",
        gradient_clipping_value: float | None = None,
        checkpoint: str | os.PathLike | Path | None = None,
        mse_loss_weight: float = 3e3,
        eikonal_loss_weight: float = 5e1,
        laplace_loss_weight: float = 1e2,
        # use_elevation: bool = False, # NOTE: switched off until we have a proper elevation dataset
        validation: float | BaseClimatrixDataset = 0.2,
        patience: int | None = None,
        overwrite_checkpoint: bool = False,
    ) -> None:
        super().__init__(dataset, target_domain)
        use_elevation = False
        if dataset.domain.is_dynamic:
            log.error("SiNET is not yet supported for dynamic datasets.")
            raise ValueError(
                "SiNET is not yet supported for dynamic datasets."
            )
        if device == "cuda" and not torch.cuda.is_available():
            log.error("CUDA is not available on this machine")
            raise ValueError("CUDA is not available on this machine")
        self.device = torch.device(device)
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
            use_elevation=use_elevation,
        )
        self.layers = layers
        self.hidden_dim = hidden_dim
        self.sorting_group_size = sorting_group_size
        self.scale = scale
        self.num_workers = num_workers
        self.num_epochs = num_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.gradient_clipping_value = gradient_clipping_value
        self.checkpoint = None
        self.is_model_loaded: bool = False

        self.mse_loss_weight = mse_loss_weight
        self.eikonal_loss_weight = eikonal_loss_weight
        self.laplace_loss_weight = laplace_loss_weight

        self.overwrite_checkpoint = overwrite_checkpoint
        if checkpoint:
            self.checkpoint = Path(checkpoint).expanduser().absolute()
            log.info("Using checkpoint path: %s", self.checkpoint)

        self.patience = patience

    @staticmethod
    def _configure_dataset_generator(
        train_coords: np.ndarray,
        train_field: np.ndarray,
        target_coords: np.ndarray,
        val_portion: float | None = None,
        val_coordinates: np.ndarray | None = None,
        val_field: np.ndarray | None = None,
        use_elevation: bool = False,
    ) -> SiNETDatasetGenerator:
        """
        Configure the SiNET dataset generator.
        """
        log.debug("Configuring SiNET dataset generator...")
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
            "target_coordinates": target_coords,
            "degree": True,
            "radius": 1.0,
            "use_elevation": use_elevation,
        }
        if val_portion is not None:
            if not (0 < val_portion < 1):
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
            kwargs["validation_coordinates"] = val_coordinates
            kwargs["validation_field"] = val_field

        return SiNETDatasetGenerator(**kwargs)

    def _configure_optimizer(
        self, siren_model: torch.nn.Module
    ) -> torch.optim.Optimizer:
        log.info(
            "Configuring Adam optimizer with learning rate: %0.6f",
            self.lr,
        )
        return torch.optim.Adam(lr=self.lr, params=siren_model.parameters())

    def _init_model(self) -> torch.nn.Module:
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

    def _maybe_clip_grads(self, siren_model: torch.nn.Module) -> None:
        if self.gradient_clipping_value:
            nn.utils.clip_grad_norm_(
                siren_model.parameters(), self.gradient_clipping_value
            )

    @log_input(log, level=logging.DEBUG)
    def _aggregate_loss(self, loss_component: LossEntity) -> torch.Tensor:
        """
        Aggregate SiNET training loss component.

        Parameters
        ----------
        loss_component : LossEntity
            The losses to be aggregated.

        Returns
        -------
        torch.Tensor
            The aggregated loss.
        """
        return (
            loss_component.mse * self.mse_loss_weight
            + loss_component.eikonal * self.eikonal_loss_weight
            + loss_component.laplace * self.laplace_loss_weight
        )

    def _maybe_load_checkpoint(
        self, siren_model: nn.Module, checkpoint: str | os.PathLike | Path
    ) -> nn.Module:
        if (
            not self.overwrite_checkpoint
            and checkpoint
            and checkpoint.exists()
        ):
            log.debug("Loading checkpoint from %s...", checkpoint)
            try:
                siren_model.load_state_dict(
                    torch.load(checkpoint, map_location=self.device)
                )
                self.is_model_loaded = True
                log.debug("Checkpoint loaded successfully.")
            except RuntimeError as e:
                log.error("Error loading checkpoint: %s.", e)
                raise e
        log.debug(
            "No checkpoint provided or checkpoint not found at %s.", checkpoint
        )
        return siren_model.to(self.device)

    def _maybe_save_checkpoint(
        self, siren_model: nn.Module, checkpoint: Path
    ) -> None:
        if checkpoint:
            if not checkpoint.parent.exists():
                log.debug(
                    "Creating checkpoint directory: %s", checkpoint.parent
                )
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
            log.debug("Saving checkpoint to %s...", checkpoint)
            try:
                torch.save(siren_model.state_dict(), checkpoint)
                log.debug("Checkpoint saved successfully.")
            except Exception as e:
                log.error("Error saving checkpoint: %s", e)
        else:
            log.debug(
                "Checkpoint saving skipped as no checkpoint path is provided."
            )

    @torch.no_grad()
    def _find_surface(self, siren_model, dataset) -> np.ndarray:
        log.debug("Finding surface using the trained INR model")
        data_loader = DataLoader(
            dataset,
            batch_size=50_000,
            shuffle=False,
        )
        all_z = []
        log.info("Creating mini-batches for surface reconstruction...")
        for i, (xy, *_) in enumerate(data_loader):
            log.info("Processing mini-batch %d/%d...", i + 1, len(data_loader))
            xy = xy.to(self.device)
            z = siren_model(xy)
            all_z.append(z.cpu().numpy())
        log.info("Surface finding complete. Concatenating results.")
        return np.concatenate(all_z)

    def _single_epoch_pass(
        self,
        data_loader: DataLoader,
        siren_model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> float:
        epoch_loss = 0.0
        for xy, true_z in data_loader:
            xy = xy.to(self.device)
            true_z = true_z.to(self.device)
            xy = xy.detach().requires_grad_(True)
            pred_z = siren_model(xy)
            loss_component: LossEntity = compute_sdf_losses(
                xy,
                pred_z * self.datasets.field_transformer.data_range_[0]
                + self.datasets.field_transformer.data_min_[0],
                true_z * self.datasets.field_transformer.data_range_[0]
                + self.datasets.field_transformer.data_min_[0],
            )
            loss = self._aggregate_loss(loss_component=loss_component)
            epoch_loss += loss.item()

            if optimizer is None:
                continue
            optimizer.zero_grad()
            loss.backward()
            self._maybe_clip_grads(siren_model)
            optimizer.step()
        return epoch_loss / len(data_loader)

    @raise_if_not_installed("torch")
    def reconstruct(self) -> BaseClimatrixDataset:
        """Reconstruct the sparse dataset using INR."""
        siren_model = self._init_model()
        siren_model = self._maybe_load_checkpoint(siren_model, self.checkpoint)

        early_stopping = EarlyStopping(
            patience=self.patience,
            delta=0.0,
            checkpoint_path=self.checkpoint,
        )
        if not self.is_model_loaded:
            optimizer = self._configure_optimizer(siren_model)
            log.info("Training SiNET model...")
            train_data_loader = DataLoader(
                self.datasets.train_dataset,
                shuffle=True,
                batch_size=self.batch_size,
                pin_memory=True,
                num_workers=self.num_workers,
            )
            val_data_loader = DataLoader(
                self.datasets.val_dataset,
                shuffle=False,
                batch_size=self.batch_size,
                pin_memory=True,
                num_workers=self.num_workers,
            )

            old_val_loss = np.inf
            for epoch in range(1, self.num_epochs + 1):
                siren_model.train()
                log.debug("Starting epoch %d/%d...", epoch, self.num_epochs)
                train_epoch_loss = self._single_epoch_pass(
                    data_loader=train_data_loader,
                    siren_model=siren_model,
                    optimizer=optimizer,
                )
                siren_model.eval()
                log.debug("Evaluating on validation set...")
                val_epoch_loss = self._single_epoch_pass(
                    data_loader=val_data_loader,
                    siren_model=siren_model,
                    optimizer=None,
                )
                log.debug(
                    "Epoch %d/%d: train loss = %0.4f | val loss = %0.4f",
                    epoch,
                    self.num_epochs,
                    train_epoch_loss,
                    val_epoch_loss,
                )
                if val_epoch_loss < old_val_loss:
                    log.debug(
                        "Validation loss improved from %0.4f to %0.4f",
                        old_val_loss,
                        val_epoch_loss,
                    )
                    self._maybe_save_checkpoint(
                        siren_model=siren_model, checkpoint=self.checkpoint
                    )
                    old_val_loss = val_epoch_loss
                if early_stopping.step(
                    val_metric=val_epoch_loss,
                    model=siren_model,
                ):
                    self._was_early_stopped = True
                    log.debug(
                        "Early stopping triggered at epoch %d/%d",
                        epoch,
                        self.num_epochs,
                    )
                    break
        siren_model.eval()
        log.info("Reconstructing target domain...")
        values = self._find_surface(siren_model, self.datasets.target_dataset)
        unscaled_values = self.datasets.field_transformer.inverse_transform(
            values
        )

        log.debug("Reconstruction completed.")
        return BaseClimatrixDataset(
            self.target_domain.to_xarray(unscaled_values, self.dataset.da.name)
        )
