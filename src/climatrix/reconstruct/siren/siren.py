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
from climatrix.reconstruct.siren.model import SIREN, SingleBVPNet

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
        sdf_loss_weight: float = 3e3,
        inter_loss_weight: float = 1e2,
        normal_loss_weight: float = 1e2,
        eikonal_loss_weight: float = 5e1,
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
            keep_aspect_ratio=False,
            num_surface_points=num_surface_points,
            num_off_surface_points=num_off_surface_points,
        )
        self.num_workers = num_workers
        self.num_epochs = num_epochs
        self.lr = lr
        self.gradient_clipping_value = gradient_clipping_value
        self.checkpoint = None
        self.is_model_loaded: bool = False
        self.sdf_loss_weight = sdf_loss_weight
        self.inter_loss_weight = inter_loss_weight
        self.normal_loss_weight = normal_loss_weight
        self.eikonal_loss_weight = eikonal_loss_weight
        if checkpoint:
            self.checkpoint = Path(checkpoint).expanduser().absolute()
            log.info("Using checkpoint path: %s", self.checkpoint)

    def _configure_optimizer(
        self, siren_model: torch.nn.Module
    ) -> torch.optim.Optimizer:
        log.info(
            "Configuring Adam optimizer with learning rate: %0.4f",
            self.lr,
        )
        return torch.optim.Adam(lr=self.lr, params=siren_model.parameters())

    def _init_model(self) -> torch.nn.Module:
        log.info("Initializing SIREN model...")
        return SingleBVPNet(type="sine", mode="mlp", in_features=3).to(
            self.device
        )

        # return SIREN(in_features=3, out_features=1, mlp=[64, 64]).to(
        # self.device
        # )

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
            loss_component.sdf * self.sdf_loss_weight
            + loss_component.inter * self.inter_loss_weight
            + loss_component.normal * self.normal_loss_weight
            + loss_component.eikonal * self.eikonal_loss_weight
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

    def _find_surface(self, siren_model, dataset) -> np.ndarray:
        log.info("Finding surface using the trained INR model")
        data_loader = DataLoader(
            dataset,
            batch_size=50_000,
            shuffle=False,
        )
        all_z = []
        log.info("Creating mini-batches for surface reconstruction...")
        for i, batch in enumerate(data_loader):
            log.info("Processing mini-batch %d/%d...", i + 1, len(data_loader))
            batch = batch.to(self.device)
            z_values = self._predict_z_values(siren_model, batch)
            all_z.append(z_values)
        log.info("Surface finding complete. Concatenating results.")
        return np.concatenate(all_z)

    @log_input(log, level=logging.DEBUG)
    def _predict_z_values(
        self,
        siren_model,
        batch,
        max_iter: int = 200,
        tol: float = 1e-6,
        alpha: float = 1e-0,
    ) -> np.ndarray:
        coordinates = batch.clone().detach().to(self.device)

        def find_temperature(decoder, x, y, z_min, z_max, num_samples=2_000):
            z_values = torch.linspace(
                z_min, z_max, num_samples, device=self.device
            )

            x_values = torch.full((num_samples,), x, device=self.device)
            y_values = torch.full((num_samples,), y, device=self.device)
            points = torch.stack([x_values, y_values, z_values], dim=1)

            with torch.no_grad():
                sdf_values = decoder(points)[1].squeeze()

            abs_sdf_values = torch.abs(sdf_values)
            min_idx = torch.argmin(abs_sdf_values)
            best_z = z_values[min_idx].item()
            return best_z

        all_z = []
        for i, row in enumerate(coordinates):
            x, y, _ = row[0].item(), row[1].item(), row[2].item()

            predicted_z = find_temperature(siren_model, x, y, -1, 1)
            all_z.append(predicted_z)

        return np.array(all_z)

        # coordinates = batch.clone().detach().to(self.device)
        # siren_model.eval()
        # for i in range(max_iter):
        #     coordinates = coordinates.detach()
        #     cc, sdf = siren_model(coordinates)

        #     if (sdf.abs() < tol).all():
        #         break
        #     grad_outputs = torch.ones_like(sdf)
        #     print("iter", i, "sdf: ", sdf.abs().sum())
        #     grads = torch.autograd.grad(sdf, [cc], grad_outputs,
        # create_graph=True)[0]
        #     coordinates[..., -1] = coordinates[..., -1]
        # - 1e-1 * (sdf / grads)[..., -1]
        # return coordinates[:, -1].detach().cpu().numpy()

        # z_values = []
        # res_sdf = []
        # for i in range(len(coordinates)):
        #     if i % 1_000 == 0:
        #         log.info(
        #             "Processing point %d/%d...", i + 1, len(coordinates)
        #         )
        #     pp = coordinates[i]

        #     for _ in range(max_iter):
        #         pp, sdf = siren_model(pp.unsqueeze(0))

        #         if (sdf.abs() < tol).all():
        #             break
        #         grad_outputs = torch.ones_like(sdf)
        #         grads = torch.autograd.grad(sdf, [pp], grad_outputs,
        # create_graph=True)[0]
        #         pp = pp.detach()
        #         pp = pp.squeeze()
        #         pp[-1] = pp[-1] - 1e-1 * (sdf / grads).squeeze()[-1]
        #     z_values.append(pp[-1].item())
        #     res_sdf.append(sdf.item())
        # return np.array(z_values)

        # grad_f_squared = 2 * sdf.view(-1, 1) * grads
        # step = grad_f_squared / (
        #     grad_f_squared.norm(dim=-1, keepdim=True) + tol
        # )

        # coordinates = coordinates.detach()
        # step = (
        #     sdf.view(-1, 1)
        #     * grads
        #     / (grads.norm(dim=-1, keepdim=True) + tol)
        # )
        # coordinates = coordinates - alpha * step

        # if (sdf.abs() < tol).all():
        #     break
        # import matplotlib.pyplot as plt
        # breakpoint()
        # coordinates = coordinates.detach().cpu().numpy().squeeze()

        return coordinates.detach().cpu().numpy().squeeze()[:, 2]

    def _form_target_coordinates(self):
        """Form target domain coordinates for reconstruction."""
        lat_grid, lon_grid = np.meshgrid(self.query_lat, self.query_lon)
        lat_grid = lat_grid.reshape(-1)
        lon_grid = lon_grid.reshape(-1)
        return np.stack(
            (lat_grid, lon_grid, np.random.uniform(-1, 1, size=len(lat_grid))),
            axis=1,
        )

    @raise_if_not_installed("torch")
    def reconstruct(self) -> DenseDataset:
        """Reconstruct the sparse dataset using INR."""
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

            for epoch in range(1, self.num_epochs + 1):
                epoch_loss = 0
                for coordinates, normals, sdf in data_loader:
                    coordinates = coordinates.to(self.device)
                    normals = normals.to(self.device)
                    sdf = sdf.to(self.device)
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
                    "Epoch %d/%d: loss = %0.4f",
                    epoch,
                    self.num_epochs,
                    epoch_loss,
                )
            self._maybe_save_checkpoint(
                siren_model=siren_model, checkpoint=self.checkpoint
            )
        siren_model.eval()
        target_grid = self._form_target_coordinates()
        target_grid_transformed = self.train_dataset.transform(target_grid)
        target_dataset = SDFPredictDataset(target_grid_transformed)
        values = self._find_surface(siren_model, target_dataset)

        # NOTE: we just transform the Z values (axis=2)
        values = self.train_dataset.inverse_transform_z(values)
        values = values.reshape(len(self.query_lon), len(self.query_lat))

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
                values.transpose(),
                coords=coordinates,
                dims=dims,
                name=self.dataset.da.name,
            )
        )
