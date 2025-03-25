from collections import namedtuple

import torch
import torch.nn.functional as F

LossEntity = namedtuple("LossEntity", ["sdf", "inter", "normal", "eikonal"])


def compute_sdf_losses(
    coords: torch.Tensor,
    pred_sdf: torch.Tensor,
    true_normals: torch.Tensor,
    true_sdf: torch.Tensor,
) -> LossEntity:

    grad_outputs = torch.ones_like(pred_sdf)
    grad = torch.autograd.grad(
        pred_sdf, [coords], grad_outputs=grad_outputs, create_graph=True
    )[0]
    sdf_constraint = torch.where(
        true_sdf != -1, pred_sdf, torch.zeros_like(pred_sdf)
    )
    inter_constraint = torch.where(
        true_sdf != -1,
        torch.zeros_like(pred_sdf),
        torch.exp(-1e2 * torch.abs(pred_sdf)),
    )
    normal_constraint = torch.where(
        true_sdf != -1,
        1 - F.cosine_similarity(grad, true_normals, dim=-1)[..., None],
        torch.zeros_like(grad[..., :1]),
    )
    grad_constraint = torch.abs(grad.norm(dim=-1) - 1)

    sdf_loss = torch.abs(sdf_constraint).mean()
    inter_loss = inter_constraint.mean()
    normal_loss = normal_constraint.mean()
    eikonal_loss = grad_constraint.mean()

    return LossEntity(
        sdf=sdf_loss,
        inter=inter_loss,
        normal=normal_loss,
        eikonal=eikonal_loss,
    )
