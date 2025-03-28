from __future__ import annotations

import logging
import os
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import xarray as xr
from torch.utils.data import DataLoader

from climatrix.decorators.runtime import log_input, raise_if_not_installed
from climatrix.reconstruct.base import BaseReconstructor
from climatrix.reconstruct.siren.dataset import (
    SDFPredictDataset,
    SDFTrainDataset,
)
from climatrix.reconstruct.siren.losses import LossEntity, compute_sdf_losses
from climatrix.reconstruct.siren.model import SingleBVPNet

if TYPE_CHECKING:
    from climatrix.dataset.dense import DenseDataset
    from climatrix.dataset.sparse import SparseDataset

log = logging.getLogger(__name__)


class SIRENReconstructor(BaseReconstructor):

    @log_input(log, level=logging.DEBUG)
    def __init__(
        self,
        dataset: SparseDataset,
        lat: slice | np.ndarray = slice(
            -90, 90, BaseReconstructor._DEFAULT_LAT_RESOLUTION
        ),
        lon: slice | np.ndarray = slice(
            -180, 180, BaseReconstructor._DEFAULT_LON_RESOLUTION
        ),
        *,
        on_surface_points: int = 1_000,
        off_surface_points: int = 1_000,
        lr: float = 1e-4,
        epochs: int = 5_000,
        num_workers: int = 0,
        device: str = "cuda",
        clip_grad: float | None = None,
        checkpoint: str | os.PathLike | Path | None = None,
    ) -> None:
        super().__init__(dataset, lat, lon)
        if dataset.is_dynamic:
            raise ValueError("SIREN is not supported for dynamic datasets.")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is not available on this machine")
        self.device = torch.device(device)
        self.dset = SDFTrainDataset(
            np.stack(
                (
                    dataset.latitude.values,
                    dataset.longitude.values,
                    dataset.da.values,
                ),
                axis=1,
            ),
            on_surface_points=on_surface_points,
            off_surface_points=off_surface_points,
            device=self.device,
        )
        self.num_workers = num_workers
        self.epoch = epochs
        self.lr = lr
        self.clip_grad = clip_grad
        self.checkpoint = None
        self.model_loaded: bool = False
        if checkpoint:
            self.checkpoint = Path(checkpoint).expanduser().absolute()

    def _configure_optimizer(
        self, model: torch.nn.Module
    ) -> torch.optim.Optimizer:
        log.debug("Configuring optimizer...")
        return torch.optim.Adam(lr=self.lr, params=model.parameters())

    def _init_model(self) -> torch.nn.Module:
        log.debug("Initializing model...")
        return SingleBVPNet(type="relu", mode="nerf", in_features=3)

    def _maybe_clip_grads(self, model: torch.nn.Module) -> None:
        if self.clip_grad:
            log.info("Clipping gradients to %0.4f...", self.clip_grad)
            nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad)

    def _aggregate_loss(self, losses: LossEntity) -> torch.Tensor:
        return (
            losses.sdf * 3e3
            + losses.inter * 1e2
            + losses.normal * 1e2
            + losses.eikonal * 5e-1
        )

    def _maybe_load_checkpoint(
        self, model: nn.Module, checkpoint: str | os.PathLike | Path
    ) -> nn.Module:
        if checkpoint and checkpoint.exists():
            log.info("Loading checkpoint from %s...", checkpoint)
            model.load_state_dict(torch.load(checkpoint))
            self.model_loaded = True
            return model
        return model

    def _maybe_save_checkpoint(
        self, model: nn.Module, checkpoint: Path
    ) -> None:
        if checkpoint and not checkpoint.exists():
            log.info("Saving checkpoint to %s...", checkpoint)
            torch.save(model.state_dict(), checkpoint)

    def _find_surface(self, model, dataset) -> np.ndarray:
        all_z = []
        data_loader = DataLoader(
            dataset,
            batch_size=5_000,
            shuffle=False,
        )
        log.debug("Creating mini-batches for surface reconstruction...")
        for i, batch in enumerate(data_loader):
            log.debug(
                "Processing mini-batch %d/%d...", i + 1, len(data_loader)
            )
            z_values = self._find_cross_point(model, batch)[:, -1]
            all_z.append(z_values)
        return np.concatenate(all_z)

    def _find_cross_point(
        self,
        model,
        batch,
        max_iter: int = 100,
        tol: float = 1e-6,
        alpha: float = 2.0,
    ) -> float:
        coords = batch.clone().detach()
        for _ in range(max_iter):
            coords, sdf = model(coords)
            grads = torch.autograd.grad(
                sdf.sum(), [coords], create_graph=True
            )[0]

            coords = coords.detach()
            step = (
                sdf.view(-1, 1)
                * grads
                / (grads.norm(dim=-1, keepdim=True) + tol)
            )
            coords[:, -1] = coords[:, -1] - alpha * step[:, -1]

            if (sdf.abs() < tol).all():
                break
        return coords.detach()

    def _form_target_coords(self):
        lat_grid, lon_grid = np.meshgrid(self.query_lat, self.query_lon)
        lat_grid = lat_grid.reshape(-1)
        lon_grid = lon_grid.reshape(-1)
        return np.stack(
            (
                lat_grid,
                lon_grid,
                np.random.normal(size=(len(lat_grid),)),
            ),
            axis=1,
        )

    @raise_if_not_installed("torch")
    def reconstruct(self) -> DenseDataset:
        from climatrix.dataset.dense import StaticDenseDataset

        model = self._init_model()
        model = self._maybe_load_checkpoint(model, self.checkpoint)
        if not self.model_loaded:
            optimizer = self._configure_optimizer(model)
            data_loader = DataLoader(
                self.dset,
                shuffle=True,
                batch_size=1,
                pin_memory=True,
                num_workers=self.num_workers,
            )

            for epoch in range(1, self.epoch + 1):
                epoch_loss = 0
                for coords, normals, sdf in data_loader:
                    coords_org, pred_sdf = model(coords)
                    losses: LossEntity = compute_sdf_losses(
                        coords_org, pred_sdf, normals, sdf
                    )
                    train_loss = self._aggregate_loss(losses=losses)
                    epoch_loss += train_loss.item()

                    optimizer.zero_grad()
                    train_loss.backward()
                    self._maybe_clip_grads(model)
                    optimizer.step()
                log.debug(
                    "Epoch %d/%d: loss = %0.4f", epoch, self.epoch, epoch_loss
                )
            self._maybe_save_checkpoint(
                model=model, checkpoint=self.checkpoint
            )
        target_dataset = SDFPredictDataset(
            self._form_target_coords(), device=self.device
        )
        # TODO: to verify finding z coordinates
        values = self._find_surface(model, target_dataset)
        values = values.reshape(len(self.query_lat), len(self.query_lon))

        coords = {
            self.dataset.latitude_name: self.query_lat,
            self.dataset.longitude_name: self.query_lon,
        }
        dims = (
            self.dataset.latitude_name,
            self.dataset.longitude_name,
        )
        log.info("Preparing StaticDenseDataset...")
        return StaticDenseDataset(
            xr.DataArray(
                values,
                coords=coords,
                dims=dims,
                name=self.dataset.da.name,
            )
        )
