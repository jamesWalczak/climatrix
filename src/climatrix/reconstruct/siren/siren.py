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
    """
    SIREN (Sinusoidal Representation Networks) reconstructor.

    Parameters
    ----------
    dataset : SparseDataset
        The input dataset.
    lat : slice or np.ndarray, optional
        The latitude range (default is slice(-90, 90, 1)).
    lon : slice or np.ndarray, optional
        The longitude range (default is slice(-180, 180, 1)).
    num_surface_points : int, optional
        The number of on-surface points (default is 1000).
    num_off_surface_points : int, optional
        The number of off-surface points (default is 1000).
    lr : float, optional
        The learning rate (default is 1e-4).
    num_epochs : int, optional
        The number of num_epochs (default is 5000).
    num_workers : int, optional
        The number of workers (default is 0).
    device : str, optional
        The device to use (default is "cuda").
    clip_grad : float, optional
        The gradient clipping value (default is None).
    checkpoint : str or os.PathLike or Path, optional
        The checkpoint file (default is None).
    """

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
        num_surface_points: int = 1_000,
        num_off_surface_points: int = 1_000,
        lr: float = 3e-4,
        num_epochs: int = 5_000,
        num_workers: int = 0,
        device: str = "cuda",
        gradient_clipping_value: float | None = None,
        checkpoint: str | os.PathLike | Path | None = None,
    ) -> None:
        super().__init__(dataset, lat, lon)
        if dataset.is_dynamic:
            log.error("SIREN is not supported for dynamic datasets.")
            raise ValueError("SIREN is not supported for dynamic datasets.")
        if device == "cuda" and not torch.cuda.is_available():
            log.error("CUDA is not available on this machine")
            raise ValueError("CUDA is not available on this machine")
        self.device = torch.device(device)
        self.train_dataset = SDFTrainDataset(
            np.stack(
                (
                    dataset.latitude.values,
                    dataset.longitude.values,
                    dataset.da.values,
                ),
                axis=1,
            ),
            num_surface_points=num_surface_points,
            num_off_surface_points=num_off_surface_points,
            device=self.device,
        )
        self.num_workers = num_workers
        self.epoch = num_epochs
        self.lr = lr
        self.gradient_clipping_value = gradient_clipping_value
        self.checkpoint = None
        self.is_model_loaded: bool = False
        if checkpoint:
            self.checkpoint = Path(checkpoint).expanduser().absolute()
            log.info("Using checkpoint path: %s", self.checkpoint)

    def _configure_optimizer(
        self, siren_model: torch.nn.Module
    ) -> torch.optim.Optimizer:
        log.info("Configuring optimizer...")
        return torch.optim.Adam(lr=self.lr, params=siren_model.parameters())

    def _init_model(self) -> torch.nn.Module:
        log.info("Initializing SIREN model...")
        return SingleBVPNet(type="relu", mode="nerf", in_features=3)

    def _maybe_clip_grads(self, siren_model: torch.nn.Module) -> None:
        if self.gradient_clipping_value:
            log.info(
                "Clipping gradients to %0.4f...", self.gradient_clipping_value
            )
            nn.utils.clip_grad_norm_(
                siren_model.parameters(), self.gradient_clipping_value
            )

    @log_input(log, level=logging.DEBUG)
    def _aggregate_loss(self, loss_component: LossEntity) -> torch.Tensor:
        """
        Aggregate SIREN training loss component.

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
            loss_component.sdf * 3e3
            + loss_component.inter * 1e2
            + loss_component.normal * 1e2
            + loss_component.eikonal * 5e-1
        )

    def _maybe_load_checkpoint(
        self, siren_model: nn.Module, checkpoint: str | os.PathLike | Path
    ) -> nn.Module:
        if checkpoint and checkpoint.exists():
            log.info("Loading checkpoint from %s...", checkpoint)
            siren_model.load_state_dict(torch.load(checkpoint))
            self.is_model_loaded = True
            return siren_model
        return siren_model

    def _maybe_save_checkpoint(
        self, siren_model: nn.Module, checkpoint: Path
    ) -> None:
        if checkpoint and not checkpoint.exists():
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            log.info("Saving checkpoint to %s...", checkpoint)
            torch.save(siren_model.state_dict(), checkpoint)

    def _find_surface(self, siren_model, dataset) -> np.ndarray:
        all_z = []
        data_loader = DataLoader(
            dataset,
            batch_size=5_000,
            shuffle=False,
        )
        log.info("Creating mini-batches for surface reconstruction...")
        for i, batch in enumerate(data_loader):
            log.info("Processing mini-batch %d/%d...", i + 1, len(data_loader))
            z_values = self._find_cross_point(siren_model, batch)[:, -1]
            all_z.append(z_values)
        return np.concatenate(all_z)

    @log_input(log, level=logging.DEBUG)
    def _find_cross_point(
        self,
        siren_model,
        batch,
        max_iter: int = 100,
        tol: float = 1e-6,
        alpha: float = 2.0,
    ) -> float:
        coordinates = batch.clone().detach()
        for _ in range(max_iter):
            coordinates, sdf = siren_model(coordinates)
            grads = torch.autograd.grad(
                sdf.sum(), [coordinates], create_graph=True
            )[0]

            coordinates = coordinates.detach()
            step = (
                sdf.view(-1, 1)
                * grads
                / (grads.norm(dim=-1, keepdim=True) + tol)
            )
            coordinates[:, -1] = coordinates[:, -1] - alpha * step[:, -1]

            if (sdf.abs() < tol).all():
                break
        return coordinates.detach()

    def _form_target_coordinates(self):
        """Form target domain coordinates for reconstruction."""
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

        siren_model = self._init_model()
        siren_model = self._maybe_load_checkpoint(siren_model, self.checkpoint)
        if not self.is_model_loaded:
            log.info("Training SIREN model...")
            optimizer = self._configure_optimizer(siren_model)
            data_loader = DataLoader(
                self.train_dataset,
                shuffle=True,
                batch_size=1,
                pin_memory=True,
                num_workers=self.num_workers,
            )

            for epoch in range(1, self.epoch + 1):
                epoch_loss = 0
                for coordinates, normals, sdf in data_loader:
                    coordinates_org, pred_sdf = siren_model(coordinates)
                    loss_component: LossEntity = compute_sdf_losses(
                        coordinates_org, pred_sdf, normals, sdf
                    )
                    train_loss = self._aggregate_loss(
                        loss_component=loss_component
                    )
                    epoch_loss += train_loss.item()

                    optimizer.zero_grad()
                    train_loss.backward()
                    self._maybe_clip_grads(siren_model)
                    optimizer.step()
                log.info(
                    "Epoch %d/%d: loss = %0.4f", epoch, self.epoch, epoch_loss
                )
            self._maybe_save_checkpoint(
                siren_model=siren_model, checkpoint=self.checkpoint
            )
        breakpoint()
        target_dataset = SDFPredictDataset(
            self.train_dataset.coordinates, device=self.device
        )
        # target_dataset = SDFPredictDataset(
        #     self._form_target_coordinates(), device=self.device
        # )
        # TODO: to verify finding z coordinates
        values = self._find_surface(siren_model, target_dataset)
        breakpoint()
        values = values.reshape(len(self.query_lat), len(self.query_lon))

        coordinates = {
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
                coordinates=coordinates,
                dims=dims,
                name=self.dataset.da.name,
            )
        )
