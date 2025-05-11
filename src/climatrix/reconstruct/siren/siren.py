from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from climatrix.dataset.domain import Domain
from climatrix.decorators.runtime import log_input, raise_if_not_installed
from climatrix.reconstruct.base import BaseReconstructor

from .losses import (
    sdf_loss,
)
from .model import SIREN

if TYPE_CHECKING:
    from climatrix.dataset.base import BaseClimatrixDataset

log = logging.getLogger(__name__)


class SIRENDataset(Dataset):

    def __init__(
        self,
        coordinates: np.ndarray,
        values: np.ndarray,
        on_surface_points: int = 1024,
    ):
        if len(values.shape) == 1:
            values = values.reshape(-1, 1)
        if values.shape[1] != 1:
            raise ValueError(
                f"Values must be shape [N] or [N, 1], got {values.shape}"
            )

        if coordinates.shape[0] != values.shape[0]:
            raise ValueError(
                f"Mismatch between coordinates ({coordinates.shape[0]})"
                f" and values ({values.shape[0]}) count"
            )

        points_3d = np.concatenate(
            [coordinates, values], axis=1
        )  # Shape [N, 3]
        log.info(
            f"Created 3D points from coordinates and values: {points_3d.shape}"
        )

        # TODO I'v added it because even though I specify nan="resample"
        # TODO a lot of values were nans
        nan_mask = np.isnan(points_3d[:, 2])
        if np.any(nan_mask):
            nan_count = np.sum(nan_mask)
            log.info(
                f"Found {nan_count} NaN values"
                f" in Z. Removing {nan_count} points."
            )
            points_3d = points_3d[~nan_mask]
            log.info(f"Points after NaN removal: {points_3d.shape}")

        self.coord_min = np.min(points_3d, axis=0)
        self.coord_max = np.max(points_3d, axis=0)

        normals_np = self._calculate_normals(points_3d)

        self.points_3d = torch.tensor(points_3d, dtype=torch.float32)
        self.normals = torch.tensor(normals_np, dtype=torch.float32)

        if self.normals.shape != self.points_3d.shape:
            log.error(
                f"Mismatch between points ({self.points_3d.shape})"
                f" and calculated normals ({self.normals.shape})"
                f" shape. Setting normals to zeros."
            )
            self.normals = torch.zeros_like(self.points_3d)

        self.points_3d = self._normalize_coords(self.points_3d)
        self.points_3d = torch.tensor(self.points_3d, dtype=torch.float32)
        log.info("Normalized 3D on-surface points.")

        self.on_surface_points = on_surface_points
        self.total_points = self.points_3d.shape[0]

        log.info(
            f"Created dataset with {self.total_points}"
            f" total on-surface points"
        )

    def _calculate_normals(self, points: np.ndarray) -> np.ndarray:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.estimate_normals()

        return np.asarray(pcd.normals)

    def _normalize_coords(self, coords_3d: torch.Tensor):
        normalized = (coords_3d - self.coord_min) / (
            self.coord_max - self.coord_min
        )
        normalized = normalized * 2.0 - 1.0
        return normalized

    def __len__(self):
        return self.total_points // self.on_surface_points

    def __getitem__(self, idx):
        """
        Return a batch of mixed on-surface and off-surface points.
        The off-surface points are generated on-the-fly in this method.
        """

        on_surface_samples = self.on_surface_points
        off_surface_samples = self.on_surface_points

        rand_idcs = torch.randint(0, self.total_points, (on_surface_samples,))
        on_surface_coords = self.points_3d[rand_idcs]
        on_surface_normals = self.normals[rand_idcs]

        off_surface_coords = torch.FloatTensor(
            off_surface_samples, 3
        ).uniform_(-1, 1)

        off_surface_normals = (
            torch.ones((off_surface_samples, 3), dtype=torch.float32) * -1
        )

        on_surface_sdf = torch.zeros(
            (on_surface_samples, 1), dtype=torch.float32
        )
        off_surface_sdf = (
            torch.ones((off_surface_samples, 1), dtype=torch.float32) * -1
        )

        coords = torch.cat([on_surface_coords, off_surface_coords], dim=0)
        normals = torch.cat([on_surface_normals, off_surface_normals], dim=0)
        sdf = torch.cat([on_surface_sdf, off_surface_sdf], dim=0)

        shuffle_idx = torch.randperm(coords.shape[0])

        return (
            coords[shuffle_idx],
            sdf[shuffle_idx],
            normals[shuffle_idx],
        )


class SIRENReconstructor(BaseReconstructor):
    @log_input(log, level=logging.DEBUG)
    def __init__(
        self,
        dataset: BaseClimatrixDataset,
        target_domain: Domain,
        *,
        on_surface_points: int = 1024,
        hidden_features: int = 256,
        hidden_layers: int = 4,
        omega_0: float = 30.0,
        omega_hidden: float = 30.0,
        lr: float = 1e-4,
        num_epochs: int = 1000,
        num_workers: int = 0,
        device: str = "cuda",
        gradient_clipping_value: float | None = None,
        checkpoint: str | os.PathLike | Path | None = None,
    ) -> None:
        super().__init__(dataset, target_domain)

        if dataset.domain.is_dynamic:
            log.error("SIREN is not yet supported for dynamic datasets.")
            raise ValueError(
                "SIREN is not yet supported for dynamic datasets."
            )

        if device == "cuda" and not torch.cuda.is_available():
            log.warning("CUDA is not available, falling back to CPU.")
            device = "cpu"

        self.on_surface_points = on_surface_points
        self.device = torch.device(device)
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.omega_0 = omega_0
        self.omega_hidden = omega_hidden
        self.lr = lr
        self.num_epochs = num_epochs
        self.num_workers = num_workers
        self.gradient_clipping_value = gradient_clipping_value
        self.is_model_loaded = False
        self.checkpoint = checkpoint

        input_coordinates_2d = dataset.domain.get_all_spatial_points()
        input_values_z = dataset.da.values.squeeze()

        self.train_dataset = SIRENDataset(
            input_coordinates_2d,
            input_values_z,
            on_surface_points=self.on_surface_points,
        )

        if checkpoint:
            self.checkpoint = Path(checkpoint).expanduser().absolute()
            log.info("Using checkpoint path: %s", self.checkpoint)

    def _init_model(self) -> nn.Module:
        log.info("Initializing 3D SIREN model")
        return SIREN(
            in_features=3,
            out_features=1,
            hidden_features=self.hidden_features,
            hidden_layers=self.hidden_layers,
            outermost_linear=True,
            omega_0=self.omega_0,
            omega_hidden=self.omega_hidden,
        ).to(self.device)

    def _configure_optimizer(self, model: nn.Module):
        log.info(
            "Configuring Adam optimizer" " with learning rate: %0.6f", self.lr
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        return optimizer

    def _maybe_clip_grads(self, model: nn.Module) -> None:
        if self.gradient_clipping_value:
            nn.utils.clip_grad_norm_(
                model.parameters(), self.gradient_clipping_value
            )

    def _maybe_load_checkpoint(
        self, model: nn.Module, checkpoint: Path | None
    ) -> nn.Module:
        """Load model weights from checkpoint if available."""
        if checkpoint and checkpoint.exists():
            log.info("Loading checkpoint from %s...", checkpoint)
            try:
                model.load_state_dict(
                    torch.load(checkpoint, map_location=self.device)
                )
                self.is_model_loaded = True
                log.info("Checkpoint loaded successfully.")
            except Exception as e:
                log.error("Error loading checkpoint: %s", e)
                log.info("Starting training from scratch.")
        else:
            log.info(
                "No checkpoint provided or checkpoint not found."
                " Starting training from scratch."
            )
        return model

    def _maybe_save_checkpoint(
        self, model: nn.Module, checkpoint: Path | None
    ) -> None:
        """Save model weights to checkpoint."""
        if checkpoint:
            if not checkpoint.parent.exists():
                log.info(
                    "Creating checkpoint directory:" " %s", checkpoint.parent
                )
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
            log.info("Saving checkpoint to %s...", checkpoint)
            try:
                torch.save(model.state_dict(), checkpoint)
                log.info("Checkpoint saved successfully.")
            except Exception as e:
                log.error("Error saving checkpoint: %s", e)
        else:
            log.info(
                "Checkpoint saving skipped as no checkpoint path is provided."
            )

    # TODO add types
    def _normalize_coords(self, coords, coord_min, coord_max):
        normalized = (coords - coord_min) / (coord_max - coord_min)
        normalized = normalized * 2.0 - 1.0
        return normalized

    # TODO add types
    def _denormalize_height(self, values, min_value, max_value):
        denormalized = (values + 1.0) / 2.0
        denormalized = denormalized * (max_value - min_value) + min_value
        return denormalized

    # TODO add types - change in the future
    def _find_closest_value(
        self, decoder, x, y, z_min, z_max, num_samples=1000
    ):
        z_values = torch.linspace(
            z_min, z_max, num_samples, device=self.device
        )

        x_values = torch.full((num_samples,), x, device=self.device)
        y_values = torch.full((num_samples,), y, device=self.device)
        points = torch.stack([x_values, y_values, z_values], dim=1)

        with torch.no_grad():
            sdf_values = decoder(points).squeeze()

        abs_sdf_values = torch.abs(sdf_values)
        min_idx = torch.argmin(abs_sdf_values)
        best_z = z_values[min_idx].item()
        return best_z

    @torch.no_grad()
    def _reconstruct_field(
        self, model: nn.Module, target_domain: Domain
    ) -> np.ndarray:
        log.info("Querying 3D SIREN model on target domain...")

        target_lat_lon_2d = (
            target_domain.get_all_spatial_points()
        )  # shape: [N, 2]
        num_spatial_points = target_lat_lon_2d.shape[0]
        target_shape = target_domain.size

        xy_min = target_lat_lon_2d.min(axis=0)
        xy_max = target_lat_lon_2d.max(axis=0)

        coord_min = self.train_dataset.coord_min
        coord_max = self.train_dataset.coord_max

        # TODO come up with better solution
        height_min = coord_min[2].item()
        height_max = coord_max[2].item()
        height_margin = (height_max - height_min) * 0.2
        query_z_min = height_min - height_margin
        query_z_max = height_max + height_margin

        normalized_xy = self._normalize_coords(
            target_lat_lon_2d, xy_min, xy_max
        )

        z_values = []
        for i in range(num_spatial_points):
            x_norm, y_norm = normalized_xy[i]

            best_z_norm = self._find_closest_value(
                model, x_norm.item(), y_norm.item(), -1, 1
            )

            best_z_real = self._denormalize_height(
                best_z_norm, query_z_min, query_z_max
            )
            z_values.append(best_z_real)

            if (i + 1) % 1000 == 0:
                log.info(f"Processed {i + 1}/{num_spatial_points} points")

        z_array = np.array(z_values).reshape(target_shape)
        return z_array

    @raise_if_not_installed("torch")
    @raise_if_not_installed("open3d")
    def reconstruct(self) -> BaseClimatrixDataset:
        from climatrix.dataset.base import BaseClimatrixDataset

        siren_model = self._init_model()
        siren_model = self._maybe_load_checkpoint(siren_model, self.checkpoint)

        if not self.is_model_loaded:
            log.info("Training 3D SIREN model (SDF loss)...")
            optimizer = self._configure_optimizer(siren_model)

            dataloader = DataLoader(
                self.train_dataset,
                batch_size=1,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
            )

            for epoch in range(1, self.num_epochs + 1):
                siren_model.train()
                epoch_loss = 0.0
                epoch_loss_components = {
                    "sdf": 0.0,
                    "inter": 0.0,
                    "normal_constraint": 0.0,
                    "grad_constraint": 0.0,
                }

                for coords_3d_normalized, gt_sdf, gt_normals in dataloader:
                    coords_3d_normalized.requires_grad_(True)

                    coords_3d_normalized = coords_3d_normalized.to(self.device)
                    gt_sdf = gt_sdf.to(self.device)
                    gt_normals = gt_normals.to(self.device)

                    pred_sdf = siren_model(coords_3d_normalized)

                    model_output = {
                        "model_in": coords_3d_normalized,
                        "model_out": pred_sdf,
                    }
                    gt_data = {"sdf": gt_sdf, "normals": gt_normals}
                    loss_dict = sdf_loss(model_output, gt_data)

                    total_loss = loss_dict["total_loss"]

                    optimizer.zero_grad()
                    total_loss.backward()
                    self._maybe_clip_grads(siren_model)
                    optimizer.step()

                    epoch_loss += total_loss.item()
                    for key in epoch_loss_components.keys():
                        if key in loss_dict:
                            epoch_loss_components[key] += loss_dict[key].item()

                if epoch % 100 == 0 or epoch == 1 or epoch == self.num_epochs:
                    avg_epoch_loss = epoch_loss / len(dataloader)
                    avg_epoch_components = {
                        k: v / len(dataloader)
                        for k, v in epoch_loss_components.items()
                    }
                    log.info(
                        f"Epoch {epoch}/{self.num_epochs}: "
                        f"Total Loss = {avg_epoch_loss:.6f}"
                    )
                    log.info(f" Loss Components: {avg_epoch_components}")

            log.info("Training finished.")
            self._maybe_save_checkpoint(siren_model, self.checkpoint)

        siren_model.eval()

        reconstructed_heights = self._reconstruct_field(
            siren_model, self.target_domain
        )

        return BaseClimatrixDataset(
            self.target_domain.to_xarray(
                reconstructed_heights, self.dataset.da.name
            )
        )
