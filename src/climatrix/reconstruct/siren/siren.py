from __future__ import annotations

from typing import TYPE_CHECKING
import os
from pathlib import Path
from itertools import product

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from climatrix.decorators.runtime import raise_if_not_installed
from climatrix.reconstruct.base import BaseReconstructor
from climatrix.reconstruct.siren.dataset import SDFDataseet
from climatrix.reconstruct.siren.losses import LossEntity, compute_sdf_losses
from climatrix.reconstruct.siren.model import SingleBVPNet

if TYPE_CHECKING:
    from climatrix.dataset.sparse import SparseDataset


class SIRENReconstructor(BaseReconstructor):

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
        on_surface_points: int = 1000,
        off_surface_points: int = 1000,
        lr: float = 1e-4,
        epochs: int = 1,  # 0_000,
        num_workers: int = 0,
        device: str = "cuda",
        clip_grad: float | None = None,
        checkpoint: str | os.PathLike | Path | None = None,
    ) -> None:
        super().__init__(dataset, lat, lon)
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is not available on this machine")
        self.device = torch.device(device)
        self.dset = SDFDataseet(
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
        return torch.optim.Adam(lr=self.lr, params=model.parameters())

    def _init_model(self) -> torch.nn.Module:
        return SingleBVPNet(type="relu", mode="nerf", in_features=3)

    def _maybe_clip_grads(self, model: torch.nn.Module) -> None:
        if self.clip_grad:
            nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad)

    def _aggregate_loss(self, losses: LossEntity) -> torch.Tensor:
        return (
            losses.sdf * 3e3
            + losses.inter * 1e2
            + losses.normal * 1e2
            + losses.eikonal * 5e-1
        )
    
    def _maybe_load_checkpoint(self, model: nn.Module, checkpoint: Path) -> nn.Module:
        if checkpoint and checkpoint.exists():
            model.load_state_dict(torch.load(checkpoint))
            return model 
        return model
    
    def _maybe_save_checkpoint(self, model: nn.Module, checkpoint: Path) -> None:
        if checkpoint and not checkpoint.exists():
            torch.save(model.state_dict(), checkpoint)

    def _find_surface(self, model) -> np.ndarray:
        lat_grid, lon_grid = np.meshgrid(self.query_lat, self.query_lon)
        coords = np.stack((lat_grid.reshape(-1), lon_grid.ravel()-1, np.zeros_like(lat_grid.ravel())), axis=1)
        all_z = []
        for *_, batch in DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(coords).float()), batch_size=40_000, shuffle=False):
            z_values = self._find_cross_point(model, batch)[:, -1]
            all_z.append(z_values)
                    
        return np.stack(all_z).reshape(lat_grid.shape)

    def _find_cross_point(self, model, batch, max_iter: int = 100, tol: float = 1e-6, alpha: float = 2.0) -> float:
        from scipy.optimize import minimize
        def func(x):
            breakpoint()
            coords = batch.clone().detach()
            coords[:,-1] = x
            coords, sdf = model(coords)
            return sdf.sum().item()

        res = minimize(func, np.zeros(len(batch)), method='nelder-mead',

               options={'xatol': 1e-8, 'disp': True})        
        # TODO: to verify and compare with 
        coords = batch.clone().detach()
        for _ in range(max_iter):
            coords, sdf = model(coords)
            grads = torch.autograd.grad(sdf.sum(), [coords], create_graph=True)[0]    
            
            coords = coords.detach()
            step = sdf.view(-1, 1) * grads / (grads.norm(dim=-1, keepdim=True) + tol)            
            coords[:,-1] = (coords[:, -1] - alpha * step[:, -1])

            if (sdf.abs() < tol).all():
                break
        return coords.detach()

    @raise_if_not_installed("torch")
    def reconstruct(self):
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
            for coords, normals, sdf in data_loader:
                coords_org, pred_sdf = model(coords)
                losses: LossEntity = compute_sdf_losses(
                    coords_org, pred_sdf, normals, sdf
                )
                train_loss = self._aggregate_loss(losses=losses)

                optimizer.zero_grad()
                train_loss.backward()
                self._maybe_clip_grads(model)
                optimizer.step()
        self._maybe_save_checkpoint(model=model, checkpoint=self.checkpoint)
        final_coords = self._find_surface(model)
        breakpoint()
        pass
