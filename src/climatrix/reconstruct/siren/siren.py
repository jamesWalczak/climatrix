from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from climatrix.dataset.domain import Domain
from climatrix.decorators.runtime import log_input, raise_if_not_installed
from climatrix.reconstruct.base import BaseReconstructor

from .dataset import SIRENDataset
from .losses import (
    LossEntity,
    sdf_loss,
)
from .model import SIREN

if TYPE_CHECKING:
    from climatrix.dataset.base import BaseClimatrixDataset

log = logging.getLogger(__name__)


class SIRENReconstructor(BaseReconstructor):
    """
    A reconstructor that uses SIREN to reconstruct fields.

    Parameters
    ----------
    dataset : BaseClimatrixDataset
        Source dataset to reconstruct from.
    target_domain : Domain
        Target domain to reconstruct onto.
    on_surface_points : int, default=1024
        Number of points to sample on the surface for training.
    hidden_features : int, default=256
        Number of features in each hidden layer.
    hidden_layers : int, default=4
        Number of hidden layers in the SIREN model.
    omega_0 : float, default=30.0
        Frequency multiplier for the first layer.
    omega_hidden : float, default=30.0
        Frequency multiplier for hidden layers.
    lr : float, default=1e-4
        Learning rate for the optimizer.
    num_epochs : int, default=100
        Number of epochs to train for.
    num_workers : int, default=0
        Number of worker processes for the dataloader.
    device : str, default="cuda"
        Device to run the model on ("cuda" or "cpu").
    gradient_clipping_value : float or None, default=None
        Value for gradient clipping (None to disable).
    checkpoint : str or os.PathLike or Path or None, default=None
        Path to save/load model checkpoint from.
    sdf_loss_weight : float, default=3000.0
        Weight for the SDF constraint loss.
    inter_loss_weight : float, default=100.0
        Weight for the interpolation consistency loss.
    normal_loss_weight : float, default=100.0
        Weight for the surface normal loss.
    grad_loss_weight : float, default=50.0
        Weight for the gradient regularization loss.

    Raises
    ------
    NotImplementedError
        If trying to use SIREN with a dynamic dataset.
    """

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
        num_epochs: int = 100,
        num_workers: int = 0,
        device: str = "cuda",
        gradient_clipping_value: float | None = None,
        checkpoint: str | os.PathLike | Path | None = None,
        sdf_loss_weight: float = 3e3,
        inter_loss_weight: float = 1e2,
        normal_loss_weight: float = 1e2,
        grad_loss_weight: float = 5e1,
    ) -> None:
        super().__init__(dataset, target_domain)

        if dataset.domain.is_dynamic:
            log.error("SIREN is not yet supported for dynamic datasets.")
            raise NotImplementedError(
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
        self.checkpoint: Path | None = None
        self.sdf_loss_weight = sdf_loss_weight
        self.inter_loss_weight = inter_loss_weight
        self.normal_loss_weight = normal_loss_weight
        self.grad_loss_weight = grad_loss_weight

        input_coordinates_2d = dataset.domain.get_all_spatial_points()
        input_values_z = dataset.da.values.flatten().squeeze()

        self.train_dataset = SIRENDataset(
            input_coordinates_2d,
            input_values_z,
            on_surface_points=self.on_surface_points,
        )

        if checkpoint:
            self.checkpoint = Path(checkpoint).expanduser().absolute()
            log.info("Using checkpoint path: %s", self.checkpoint)

    def _init_model(self) -> nn.Module:
        """
        Initialize the 3D SIREN model.

        Returns
        -------
        nn.Module
            Initialized SIREN model on the appropriate device.
        """
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

    def _configure_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """
        Configure the optimizer for the model.

        Parameters
        ----------
        model : nn.Module
            The model to optimize.

        Returns
        -------
        torch.optim.Optimizer
            Configured Adam optimizer.
        """
        log.info(
            "Configuring Adam optimizer" " with learning rate: %0.6f", self.lr
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        return optimizer

    def _maybe_clip_grads(self, model: nn.Module) -> None:
        """
        Apply gradient clipping to model parameters if configured.

        Parameters
        ----------
        model : nn.Module
            The model whose gradients may need clipping.
        """
        if self.gradient_clipping_value:
            nn.utils.clip_grad_norm_(
                model.parameters(), self.gradient_clipping_value
            )

    @log_input(log, level=logging.DEBUG)
    def _apply_weights_to_loss_components(
        self, loss_component: LossEntity
    ) -> None:
        """
        Scales each loss component in the input LossEntity by its weight.

        Parameters
        ----------
        loss_component : LossEntity
            Contains sdf, inter, normal, and grad loss components.
        """
        loss_component.sdf *= self.sdf_loss_weight
        loss_component.inter *= self.inter_loss_weight
        loss_component.normal *= self.normal_loss_weight
        loss_component.grad *= self.grad_loss_weight

    def _maybe_load_checkpoint(
        self, model: nn.Module, checkpoint: Path | None
    ) -> nn.Module:
        """
        Load model weights from checkpoint if available.

        Parameters
        ----------
        model : nn.Module
            The model to load weights into.
        checkpoint : Path or None
            Path to the checkpoint file.

        Returns
        -------
        nn.Module
            The model, potentially with loaded weights.
        """
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
        """
        Save model weights to checkpoint.

        Parameters
        ----------
        model : nn.Module
            The model to save weights from.
        checkpoint : Path or None
            Path to save the checkpoint to.
        """
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

    def _normalize_coords(
        self, coords: np.ndarray, coord_min: np.ndarray, coord_max: np.ndarray
    ) -> np.ndarray:
        """
        Normalize coordinates to the range [-1, 1].

        Parameters
        ----------
        coords : np.ndarray
            Coordinates to normalize.
        coord_min : np.ndarray
            Minimum coordinate values.
        coord_max : np.ndarray
            Maximum coordinate values.

        Returns
        -------
        np.ndarray
            Normalized coordinates in range [-1, 1].
        """
        normalized = (coords - coord_min) / (coord_max - coord_min)
        normalized = normalized * 2.0 - 1.0
        return normalized

    def _denormalize_values(
        self, values: float | np.ndarray, min_value: float, max_value: float
    ) -> float | np.ndarray:
        """
        Denormalize values from [-1, 1] to original range.

        Parameters
        ----------
        values : float or np.ndarray
            Normalized values to denormalize.
        min_value : float
            Minimum value in original range.
        max_value : float
            Maximum value in original range.

        Returns
        -------
        float or np.ndarray
            Denormalized values in original range.
        """
        denormalized = (values + 1.0) / 2.0
        denormalized = denormalized * (max_value - min_value) + min_value
        return denormalized

    def _find_closest_value(
        self,
        decoder: nn.Module,
        x: float,
        y: float,
        z_min: float,
        z_max: float,
        num_samples: int = 1000,
    ) -> float:
        """
        Find the z value where the SDF is closest
        to zero for given x, y coordinates.

        Parameters
        ----------
        decoder : nn.Module
            The SIREN model to query.
        x : float
            Normalized x coordinate.
        y : float
            Normalized y coordinate.
        z_min : float
            Minimum normalized z value to sample.
        z_max : float
            Maximum normalized z value to sample.
        num_samples : int, default=1000
            Number of z values to sample.

        Returns
        -------
        float
            The z value where SDF is closest to zero.
        """
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
        """
        Reconstruct the field on the target domain
        using the trained SIREN model.

        Parameters
        ----------
        model : nn.Module
            Trained SIREN model.
        target_domain : Domain
            Target domain to reconstruct onto.

        Returns
        -------
        np.ndarray
            Reconstructed field values.
        """
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

        height_min = coord_min[2].item()
        height_max = coord_max[2].item()
        height_margin = (height_max - height_min) * 0.2
        query_z_min = height_min - height_margin
        query_z_max = height_max + height_margin

        normalized_xy = self._normalize_coords(
            target_lat_lon_2d, xy_min, xy_max
        )

        z_values: list[float] = []
        for i in range(num_spatial_points):
            x_norm, y_norm = normalized_xy[i]

            best_z_norm = self._find_closest_value(
                model, x_norm.item(), y_norm.item(), -1, 1
            )

            best_z_real = self._denormalize_values(
                best_z_norm, query_z_min, query_z_max
            )
            z_values.append(best_z_real)

            if (i + 1) % 1000 == 0:
                log.info(f"Processed {i + 1}/{num_spatial_points} points")

        z_array = np.array(z_values).reshape(target_shape)
        return z_array

    @raise_if_not_installed("torch")
    def reconstruct(self) -> BaseClimatrixDataset:
        """
        Train (if necessary) and use a SIREN model to reconstruct the field.

        This method is the main entry point for using the SIREN reconstructor.
        It will train a new model if no checkpoint was loaded, and then use the
        model to reconstruct the field on the target domain.

        Returns
        -------
        BaseClimatrixDataset
            A dataset containing the reconstructed field.

        Raises
        ------
        ImportError
            If required dependencies are not installed.
        """
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
                epoch_loss_components: dict[str, float] = {
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

                    model_output: dict[str, torch.Tensor] = {
                        "model_in": coords_3d_normalized,
                        "model_out": pred_sdf,
                    }
                    gt_data: dict[str, torch.Tensor] = {
                        "sdf": gt_sdf,
                        "normals": gt_normals,
                    }
                    loss_component: LossEntity = sdf_loss(
                        model_output, gt_data
                    )

                    self._apply_weights_to_loss_components(loss_component)

                    train_loss = (
                        loss_component.sdf
                        + loss_component.inter
                        + loss_component.normal
                        + loss_component.grad
                    )

                    optimizer.zero_grad()
                    train_loss.backward()
                    self._maybe_clip_grads(siren_model)
                    optimizer.step()

                    epoch_loss += train_loss.item()
                    epoch_loss_components["sdf"] += loss_component.sdf.item()
                    epoch_loss_components[
                        "inter"
                    ] += loss_component.inter.item()
                    epoch_loss_components[
                        "normal_constraint"
                    ] += loss_component.normal.item()
                    epoch_loss_components[
                        "grad_constraint"
                    ] += loss_component.grad.item()

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

        reconstructed_values = self._reconstruct_field(
            siren_model, self.target_domain
        )
        return BaseClimatrixDataset(
            self.target_domain.to_xarray(
                reconstructed_values, self.dataset.da.name
            )
        )
