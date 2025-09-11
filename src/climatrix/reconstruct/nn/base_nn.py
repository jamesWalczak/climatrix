import logging
import os
from abc import abstractmethod
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from climatrix.dataset.base import BaseClimatrixDataset
from climatrix.dataset.domain import Domain
from climatrix.reconstruct.base import BaseReconstructor, Hyperparameter
from climatrix.reconstruct.nn.callbacks import EarlyStopping
from climatrix.reconstruct.nn.dataset import BaseNNDatasetGenerator

log = logging.getLogger(__name__)


class BaseNNReconstructor(BaseReconstructor):
    """
    Base class for neural network-based reconstructor.
    This class provides a common interface for all neural network-based reconstruction methods.
    """

    dataset_generator_type: ClassVar[type[BaseNNDatasetGenerator]]

    is_model_loaded: bool = False
    checkpoint: Path | None = None
    gradient_clipping_value: float | None = None
    overwrite_checkpoint: bool = False
    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    num_workers: int
    patience: int | None = None
    datasets: BaseNNDatasetGenerator
    validation: float | BaseClimatrixDataset | None = None
    _custom_dataset_generator_kwargs: dict

    # Hyperparameters definitions
    lr = Hyperparameter[float](
        bounds=(np.finfo(float).eps, None), default=5e-3
    )
    num_epochs = Hyperparameter[int](bounds=(1, None), default=1_000)
    batch_size = Hyperparameter[int](bounds=(1, None), default=32)
    weight_decay = Hyperparameter[float](bounds=(0.0, None), default=0.0)

    def __init__(
        self,
        dataset: BaseClimatrixDataset,
        target_domain: Domain,
        lr: float,
        num_epochs: int,
        batch_size: int,
        weight_decay: float | None = None,
        checkpoint: str | os.PathLike | Path | None = None,
        overwrite_checkpoint: bool = False,
        num_workers: int = 0,
        gradient_clipping_value: float | None = None,
        device: str = "cuda",
        patience: int | None = None,
        validation: float | BaseClimatrixDataset | None = None,
    ) -> None:
        super().__init__(dataset, target_domain)
        self.num_workers = num_workers
        self.patience = patience
        self.checkpoint = None
        self.is_model_loaded: bool = False
        self.overwrite_checkpoint = overwrite_checkpoint
        self.gradient_clipping_value = gradient_clipping_value
        self.device = torch.device(device)
        if checkpoint:
            self.checkpoint = Path(checkpoint).expanduser().absolute()
            log.info("Using checkpoint path: %s", self.checkpoint)

        self.lr = lr
        if weight_decay is not None:
            self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self._custom_dataset_generator_kwargs = {}
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
        )

    def _maybe_clip_grads(self, nn_model: torch.nn.Module) -> None:
        if self.gradient_clipping_value:
            nn.utils.clip_grad_norm_(
                nn_model.parameters(), self.gradient_clipping_value
            )

    def _maybe_load_checkpoint(
        self, nn_model: nn.Module, checkpoint: str | os.PathLike | Path | None
    ) -> nn.Module:
        if (
            not self.overwrite_checkpoint
            and checkpoint
            and Path(checkpoint).exists()
        ):
            log.debug("Loading checkpoint from %s...", checkpoint)
            try:
                nn_model.load_state_dict(
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
        return nn_model.to(self.device)

    def _maybe_save_checkpoint(
        self, nn_model: nn.Module, checkpoint: Path | None
    ) -> None:
        if checkpoint:
            if not checkpoint.parent.exists():
                log.debug(
                    "Creating checkpoint directory: %s", checkpoint.parent
                )
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
            log.debug("Saving checkpoint to %s...", checkpoint)
            try:
                torch.save(nn_model.state_dict(), checkpoint)
                log.debug("Checkpoint saved successfully.")
            except Exception as e:
                log.error("Error saving checkpoint: %s", e)
        else:
            log.debug(
                "Checkpoint saving skipped as no checkpoint path is provided."
            )

    @torch.no_grad()
    def _find_surface(
        self, nn_model, dataset, batch_size: int = 50_000
    ) -> np.ndarray:
        log.debug("Finding surface using the trained INR model")
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
        )
        all_z = []
        log.info("Creating mini-batches for surface reconstruction...")
        for i, (xy, *_) in enumerate(data_loader):
            log.info("Processing mini-batch %d/%d...", i + 1, len(data_loader))
            xy = xy.to(self.device)
            z = nn_model(xy)
            all_z.append(z.cpu().numpy())
        log.info("Surface finding complete. Concatenating results.")
        return np.concatenate(all_z)

    def _single_epoch_pass(
        self,
        data_loader: DataLoader,
        nn_model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> float:
        epoch_loss = 0.0
        for xy, true_z in data_loader:
            xy = xy.to(self.device)
            true_z = true_z.to(self.device)
            xy = xy.detach().requires_grad_(True)
            pred_z = nn_model(xy)
            loss = self.compute_loss(xy, pred_z, true_z)
            epoch_loss += loss.item()

            if optimizer is None:
                continue
            optimizer.zero_grad()
            loss.backward()
            self._maybe_clip_grads(nn_model)
            optimizer.step()
        return epoch_loss / len(data_loader)

    def reconstruct(self):
        nn_model = self.init_model()
        nn_model = self._maybe_load_checkpoint(nn_model, self.checkpoint)

        early_stopping = EarlyStopping(
            patience=self.patience,
            delta=0.0,
            checkpoint_path=self.checkpoint,
        )
        if not self.is_model_loaded:
            optimizer = self.configure_optimizer(nn_model)
            epoch_schedulers = self.configure_epoch_schedulers(optimizer)
            if epoch_schedulers:
                log.info("Configuring epoch schedulers...")
            log.info("Training %s model...", type(nn_model).__name__)
            train_data_loader = DataLoader(
                self.datasets.train_dataset,
                shuffle=True,
                batch_size=self.batch_size,
                pin_memory=True,
                num_workers=self.num_workers,
            )
            if self.datasets.val_dataset is not None:
                log.info(
                    "Validation dataset is available. Using it for validation."
                )
                val_data_loader = DataLoader(
                    self.datasets.val_dataset,
                    shuffle=False,
                    batch_size=self.batch_size,
                    pin_memory=True,
                    num_workers=self.num_workers,
                )
            else:
                log.warning(
                    "Validation dataset is not available. Skipping validation."
                )
                val_data_loader = None

            old_val_loss = np.inf
            for epoch in range(1, self.num_epochs + 1):
                nn_model.train()
                log.debug("Starting epoch %d/%d...", epoch, self.num_epochs)
                train_epoch_loss = self._single_epoch_pass(
                    data_loader=train_data_loader,
                    nn_model=nn_model,
                    optimizer=optimizer,
                )

                for scheduler in epoch_schedulers:
                    scheduler.step()

                # NOTE: we do not use `with torch.no_grad()` here
                # because we want to compute gradients for losses
                if val_data_loader is None:
                    continue
                nn_model.eval()
                log.debug("Evaluating on validation set...")
                val_epoch_loss = self._single_epoch_pass(
                    data_loader=val_data_loader,
                    nn_model=nn_model,
                    optimizer=None,
                )
                log.info(
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
                        nn_model=nn_model, checkpoint=self.checkpoint
                    )
                    old_val_loss = val_epoch_loss

                if early_stopping.step(
                    val_metric=val_epoch_loss,
                    model=nn_model,
                ):
                    self._was_early_stopped = True
                    log.debug(
                        "Early stopping triggered at epoch %d/%d",
                        epoch,
                        self.num_epochs,
                    )
                    break
        nn_model.eval()

        log.info("Reconstructing target domain...")
        values = self._find_surface(nn_model, self.datasets.target_dataset)
        unscaled_values = self.datasets.untransform_field(values)

        log.debug("Reconstruction completed.")
        name = str(self.dataset.da.name) or "reconstructed_field"

        return BaseClimatrixDataset(
            self.target_domain.to_xarray(unscaled_values, name)
        )

    @abstractmethod
    def compute_loss(
        self, xy: torch.Tensor, pred_z: torch.Tensor, true_z: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the loss for the given inputs and predictions.
        """
        raise NotImplementedError(
            "Subclasses must implement compute_loss method."
        )

    @abstractmethod
    def init_model(self) -> nn.Module:
        """
        Initialize the neural network model.
        """
        raise NotImplementedError(
            "Subclasses must implement init_model method."
        )

    @abstractmethod
    def configure_optimizer(
        self, nn_model: nn.Module
    ) -> torch.optim.Optimizer:
        """
        Configure the optimizer for the neural network model.
        """
        raise NotImplementedError(
            "Subclasses must implement configure_optimizer method."
        )

    def _configure_dataset_generator(
        self,
        train_coords: np.ndarray,
        train_field: np.ndarray,
        target_coords: np.ndarray,
        val_portion: float | None = None,
        val_coordinates: np.ndarray | None = None,
        val_field: np.ndarray | None = None,
    ) -> BaseNNDatasetGenerator:
        log.debug("Configuring dataset generator...")
        if val_portion is not None and (
            val_coordinates is not None or val_field is not None
        ):
            log.error(
                "Cannot use both `val_portion` and `val_coordinates`/`val_field`."
            )
            raise ValueError(
                "Cannot use both `val_portion` and `val_coordinates`/`val_field`."
            )
        kwargs: dict[str, Any] = {
            "spatial_points": train_coords,
            "field": train_field,
            "target_coordinates": target_coords,
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

        kwargs.update(self._custom_dataset_generator_kwargs)
        return self.dataset_generator_type(**kwargs)

    def configure_epoch_schedulers(
        self, optimizer: torch.optim.Optimizer
    ) -> list[torch.optim.lr_scheduler.LRScheduler]:
        """
        Configure the epoch schedulers for the optimizer.
        """
        return []
