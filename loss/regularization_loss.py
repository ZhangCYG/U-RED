import torch
import torch.nn.functional as F
import pytorch3d.ops as ops

def regularization_m(matching_m):
    # matching_m: bs x 1024 x 2048
    max_m, _ = torch.max(matching_m, dim=-1)
    loss = torch.mean(max_m.view(-1))
    return loss


def regularization_m_distribute(matching_m, x, x_source):
    # matching_m: typically bs x 1024 x 2048
    # x is the input point, bs x numpoints x 3, x_source is bs x 2048 x 3
    # find the nearest neighbor of x and x_source

    nofsource = x_source.shape[1]
    nofinput = x.shape[1]
    x_repeat = x.unsqueeze(2).repeat(1, 1, nofsource, 1)
    x_source_repeat = x_source.unsqueeze(1).repeat(1, nofinput, 1, 1)
    res_m = torch.abs(x_repeat - x_source_repeat)
    res_m = torch.norm(res_m, dim=-1)  # bs x 1024 x 2048
    loss = torch.mean(torch.square(torch.sum(matching_m * res_m, dim=-1)))  # bs x nofinput
    return loss


def regularization_re_residuals(re_residuals):
    # residuals: bs x 1024 x 3
    # L1 residuals
    residual_loss_reg = torch.mean(torch.sum(torch.abs(re_residuals), dim=-1))
    return residual_loss_reg


def regularization_m_spd(matching_m):
    # matching m should not be softmaxed
    soft_m = F.softmax(matching_m, dim=-1)
    log_soft = F.log_softmax(matching_m, dim=-1)
    entropy_loss = torch.mean(-torch.sum(soft_m * log_soft, dim=-1))
    return entropy_loss


def regularization_m_divergent(matching_m):
    # to use as many points in source as possible
    # matching_m   bs x nnofpoint x source points
    matching_list, _ = torch.max(matching_m, dim=1)  # bs x 2048
    loss = torch.mean(matching_list)
    return loss
