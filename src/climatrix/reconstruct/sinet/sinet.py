from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import xarray as xr
from torch.utils.data import DataLoader

from climatrix import BaseClimatrixDataset, Domain
from climatrix.decorators.runtime import log_input, raise_if_not_installed
from climatrix.reconstruct.base import BaseReconstructor
from climatrix.reconstruct.sinet.dataset import (
    SiNETDatasetGenerator,
)
from climatrix.reconstruct.sinet.losses import LossEntity, compute_sdf_losses
from climatrix.reconstruct.sinet.model import SiNET

log = logging.getLogger(__name__)


class SiNETReconstructor(BaseReconstructor):

    @log_input(log, level=logging.DEBUG)
    def __init__(
        self,
        dataset: BaseClimatrixDataset,
        target_domain: Domain,
        *,
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
    ) -> None:
        super().__init__(dataset, target_domain)
        if dataset.is_dynamic:
            log.error("SiNET is not yet supported for dynamic datasets.")
            raise ValueError(
                "SiNET is not yet supported for dynamic datasets."
            )
        if device == "cuda" and not torch.cuda.is_available():
            log.error("CUDA is not available on this machine")
            raise ValueError("CUDA is not available on this machine")
        self.device = torch.device(device)
        self.datasets = SiNETDatasetGenerator(
            dataset.latitude.values,
            dataset.longitude.values,
            dataset.da.values,
            degree=True,
            radius=1,
        )
        dense_query_lat, dense_query_lon = self._form_query_pairs()
        self.datasets.set_target_coordinates(
            dense_query_lat, dense_query_lon, degree=True
        )
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

        if checkpoint:
            self.checkpoint = Path(checkpoint).expanduser().absolute()
            log.info("Using checkpoint path: %s", self.checkpoint)

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
        # NOTE: we are using 3 input cooridnates as lat/lon are converted
        # to cartesian coordinates on unit sphere
        return SiNET(in_features=2, out_features=1, mlp=[256, 256, 256]).to(
            self.device
        )

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
        if checkpoint and checkpoint.exists():
            log.info("Loading checkpoint from %s...", checkpoint)
            try:
                siren_model.load_state_dict(
                    torch.load(checkpoint, map_location=self.device)
                )
                self.is_model_loaded = True
                log.info("Checkpoint loaded successfully.")
            except RuntimeError as e:
                log.error("Error loading checkpoint: %s", e)
        log.info(
            "No checkpoint provided or checkpoint not found at %s.", checkpoint
        )
        return siren_model.to(self.device)

    def _maybe_save_checkpoint(
        self, siren_model: nn.Module, checkpoint: Path
    ) -> None:
        if checkpoint:
            if not checkpoint.parent.exists():
                log.info(
                    "Creating checkpoint directory: %s", checkpoint.parent
                )
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
            log.info("Saving checkpoint to %s...", checkpoint)
            try:
                torch.save(siren_model.state_dict(), checkpoint)
                log.info("Checkpoint saved successfully.")
            except Exception as e:
                log.error("Error saving checkpoint: %s", e)
        else:
            log.info(
                "Checkpoint saving skipped as no checkpoint path is provided."
            )

    @torch.no_grad()
    def _find_surface(self, siren_model, dataset) -> np.ndarray:
        log.info("Finding surface using the trained INR model")
        data_loader = DataLoader(
            dataset,
            batch_size=50_000,
            shuffle=False,
        )
        all_z = []
        log.info("Creating mini-batches for surface reconstruction...")
        for i, xy in enumerate(data_loader):
            log.info("Processing mini-batch %d/%d...", i + 1, len(data_loader))
            xy = xy[0].to(self.device)
            z = siren_model(xy)
            all_z.append(z.cpu().numpy())
        log.info("Surface finding complete. Concatenating results.")
        return np.concatenate(all_z)

    def _form_query_pairs(self) -> tuple[np.ndarray, np.ndarray]:
        """Form target domain coordinates for reconstruction."""
        lat_grid, lon_grid = np.meshgrid(self.query_lat, self.query_lon)
        return lat_grid.reshape(-1), lon_grid.reshape(-1)

    @raise_if_not_installed("torch")
    def reconstruct(self) -> BaseClimatrixDataset:
        """Reconstruct the sparse dataset using INR."""
        siren_model = self._init_model()
        siren_model = self._maybe_load_checkpoint(siren_model, self.checkpoint)
        if not self.is_model_loaded:
            log.info("Training SiNET model...")
            optimizer = self._configure_optimizer(siren_model)
            data_loader = DataLoader(
                self.datasets.train_dataset,
                shuffle=True,
                batch_size=self.batch_size,
                pin_memory=True,
                num_workers=self.num_workers,
            )

            for epoch in range(1, self.num_epochs + 1):
                epoch_loss = 0
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
                    # loss_component: LossEntity = compute_sdf_losses(
                    #     xy, pred_z, true_z
                    # )
                    train_loss = self._aggregate_loss(
                        loss_component=loss_component
                    )
                    epoch_loss += train_loss.item()

                    optimizer.zero_grad()
                    train_loss.backward()
                    self._maybe_clip_grads(siren_model)
                    optimizer.step()
                log.info(
                    "Epoch %d/%d: loss = %0.4f",
                    epoch,
                    self.num_epochs,
                    epoch_loss,
                )
            self._maybe_save_checkpoint(
                siren_model=siren_model, checkpoint=self.checkpoint
            )
        siren_model.eval()
        # values = self._find_surface(siren_model, self.datasets.target_dataset)
        # unscaled_values = values
        # unscaled_values = self.datasets.field_transformer.inverse_transform(
        #     values
        # )

        # coordinates = {
        #     self.dataset.latitude.name: self.query_lat,
        #     self.dataset.longitude.name: self.query_lon,
        # }
        # dims = (
        #     self.dataset.latitude.name,
        #     self.dataset.longitude.name,
        # )
        log.info("Preparing StaticDenseDataset...")
        raise NotImplementedError()
