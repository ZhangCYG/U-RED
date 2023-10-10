from pytorch3d.loss import chamfer_distance
import torch


def compute_cm_loss(source_p, target_p, batch_reduction="mean"):
    loss, _ = chamfer_distance(source_p, target_p, batch_reduction=batch_reduction)
    return loss
