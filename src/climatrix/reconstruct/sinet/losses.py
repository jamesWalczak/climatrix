from collections import namedtuple

import torch
import torch.nn.functional as F

LossEntity = namedtuple("LossEntity", ["mse", "eikonal", "laplace"])


def compute_sdf_losses(
    xy: torch.Tensor, pred_z: torch.Tensor, true_z: torch.Tensor
) -> LossEntity:
    pred_z = pred_z.squeeze()
    true_z = true_z.squeeze()
    mse_loss = F.mse_loss(pred_z, true_z)

    grad_outputs = torch.ones_like(pred_z)
    grad = torch.autograd.grad(
        pred_z, [xy], grad_outputs=grad_outputs, create_graph=True
    )[0]
    eikonal_loss = torch.abs(grad.norm(dim=-1) - 1).mean()

    grad2 = torch.autograd.grad(grad.sum(), [xy], create_graph=True)[0]
    laplace_loss = (grad2**2).norm(dim=-1).mean()

    return LossEntity(mse=mse_loss, eikonal=eikonal_loss, laplace=laplace_loss)
