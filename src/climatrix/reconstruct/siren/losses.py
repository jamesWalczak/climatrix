import torch
import torch.nn.functional as F


def gradient(y, x):
    y = y.squeeze()
    dydx = torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
    )[0]
    return dydx


def sdf_loss(model_output, gt):
    gt_sdf = gt["sdf"]
    gt_normals = gt["normals"]

    coords = model_output["model_in"]
    pred_sdf = model_output["model_out"]

    if gt_sdf.shape != pred_sdf.shape:
        pred_sdf = pred_sdf.squeeze(-1)[..., None]
        gt_sdf = gt_sdf.squeeze(-1)[..., None]

    pred_gradient = gradient(pred_sdf, coords)

    sdf_constraint_active = torch.where(
        gt_sdf != -1, pred_sdf - gt_sdf, torch.zeros_like(pred_sdf)
    )
    sdf_loss_val = torch.abs(sdf_constraint_active).mean()

    inter_constraint_active = torch.where(
        gt_sdf == -1,
        torch.exp(-1e2 * torch.abs(pred_sdf)),
        torch.zeros_like(pred_sdf),
    )
    inter_loss_val = inter_constraint_active.mean()

    normal_constraint_val = torch.where(
        gt_sdf != -1,
        1 - F.cosine_similarity(pred_gradient, gt_normals, dim=-1)[..., None],
        torch.zeros_like(pred_gradient[..., :1]),
    )[..., None].mean()

    grad_constraint_val = torch.abs(pred_gradient.norm(dim=-1) - 1).mean()

    weighted_sdf_loss = sdf_loss_val * 3e3
    weighted_inter_loss = inter_loss_val * 1e2
    weighted_normal_loss = normal_constraint_val * 1e2
    weighted_grad_loss = grad_constraint_val * 5e1

    loss = (
        weighted_sdf_loss
        + weighted_inter_loss
        + weighted_normal_loss
        + weighted_grad_loss
    )

    return {
        "total_loss": loss,
        "sdf": weighted_sdf_loss,
        "inter": weighted_inter_loss,
        "normal_constraint": weighted_normal_loss,
        "grad_constraint": weighted_grad_loss,
    }
