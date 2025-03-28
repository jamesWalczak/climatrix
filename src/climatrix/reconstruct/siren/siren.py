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
from climatrix.reconstruct.siren.dataset import SDFDataset
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
        epochs: int = 500,
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
        self.dset = SDFDataset(
            dataset,
            on_surface_points=on_surface_points,
            off_surface_points=off_surface_points,
            device=self.device,
        )
        self.num_workers = num_workers
        self.epoch = epochs
        self.lr = lr
        self.clip_grad = clip_grad
        self.checkpoint = None
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
        self, model: nn.Module, checkpoint: Path
    ) -> nn.Module:
        if checkpoint and checkpoint.exists():
            log.info("Loading checkpoint from %s...", checkpoint)
            model.load_state_dict(torch.load(checkpoint))
            return model
        return model

    def _maybe_save_checkpoint(
        self, model: nn.Module, checkpoint: Path
    ) -> None:
        if checkpoint and not checkpoint.exists():
            log.info("Saving checkpoint to %s...", checkpoint)
            torch.save(model.state_dict(), checkpoint)

    def _find_surface(self, model) -> np.ndarray:
        from scipy.optimize import minimize

        def func(x):
            coords = batch.clone().detach()
            coords[:, -1] = torch.from_numpy(x)
            coords, sdf = model(coords)
            return sdf.abs().sum().item()

        lat_grid, lon_grid = np.meshgrid(self.query_lat, self.query_lon)
        coords = np.stack(
            (
                lat_grid.reshape(-1),
                lon_grid.ravel() - 1,
                np.zeros_like(lat_grid.ravel()),
            ),
            axis=1,
        )
        all_z = []
        # breakpoint()
        log.debug("Creating mini-batches for surface reconstruction...")
        for *_, batch in DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(coords).float()),
            batch_size=5_000,
            shuffle=False,
        ):
            z_values = self._find_cross_point(model, batch)[:, -1]
            all_z.append(z_values)
            # res = minimize(func, np.zeros(len(batch)), method='Powell',options={'disp': True})
            # all_z.append(res.x)
        return np.concatenate(all_z).reshape(lat_grid.shape)

    def _find_cross_point(
        self,
        model,
        batch,
        max_iter: int = 100,
        tol: float = 1e-6,
        alpha: float = 2.0,
    ) -> float:
        from scipy.optimize import minimize

        def func(x):
            coords = batch.clone().detach()
            coords[:, -1] = torch.from_numpy(x)
            coords, sdf = model(coords)
            return sdf.abs().sum().item()

        # breakpoint()
        # res = minimize(func, np.zeros(len(batch)), method='Powell',options={'disp': True})
        # TODO: to verify and compare with
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

    @raise_if_not_installed("torch")
    def reconstruct(self) -> DenseDataset:
        from climatrix.dataset.dense import StaticDenseDataset

        model = self._init_model()
        model = self._maybe_load_checkpoint(model, self.checkpoint)
        optimizer = self._configure_optimizer(model)
        data_loader = DataLoader(
            self.dset,
            shuffle=True,
            batch_size=1,
            pin_memory=True,
            num_workers=self.num_workers,
        )

        for epoch in range(self.epoch):
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
        self._maybe_save_checkpoint(model=model, checkpoint=self.checkpoint)
        values = self._find_surface(model)

        coords = {
            self.dataset.latitude_name: self.query_lat,
            self.dataset.longitude_name: self.query_lon,
        }
        dims = (
            self.dataset.longitude_name,
            self.dataset.latitude_name,
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
