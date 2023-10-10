import torch
import torch.nn.functional as F
import pytorch3d.ops as ops


# using matching_m to transfer the deformed shape to the target
# then point-wise loss
# I use square root distance as the loss here.
# in chamfer distance, the loss is square loss
# I found that use square root loss can help the matching network converge to a smaller value
def point_loss_matching(deformed_p, matching_m, target_p):
    # deformed_p, bs x 2048 x 3
    # matching_m, bs x 1024 x 2048
    # target_p, bs x 1024 x 3
    target_p_0 = torch.matmul(matching_m, deformed_p)
    res_p = target_p_0 - target_p
    res_p = torch.abs(res_p)
    loss = torch.mean(res_p)
    return loss


# square loss
def point_loss_matching_no_mean(deformed_p, matching_m, target_p):
    # deformed_p, bs x 2048 x 3
    # matching_m, bs x 1024 x 2048
    # target_p, bs x 1024 x 3
    target_p_0 = torch.matmul(matching_m, deformed_p)
    res_p = target_p_0 - target_p
    res_p_2 = torch.sum(res_p * res_p, dim=-1)  # bs x 1024
    loss = torch.mean(res_p_2, dim=-1)  # bs
    return loss


def retrieval_regression_loss(emd_dis, actual_dis, obj_sigmas):
    obj_sigmas = torch.sigmoid(obj_sigmas)

    emd_dis = emd_dis / 10.0
    qij = F.softmax(-emd_dis, dim=0)

    actual_dis = actual_dis

    pij = torch.div(actual_dis, 1.0)
    pij = F.softmax(-pij, dim=0)

    loss = torch.sum(torch.abs(pij - qij), dim=0)
    loss_part_2 = torch.sum(torch.abs(emd_dis - actual_dis), dim=0)

    return loss + 0.05 * loss_part_2


def retrieval_regression_loss2(emd_dis, actual_dis, obj_sigmas):
    obj_sigmas = torch.sigmoid(obj_sigmas)

    emd_dis = emd_dis / 10.0
    loss_part_2 = torch.sum(torch.abs(emd_dis - actual_dis), dim=0)

    return loss_part_2


def retrieval_regression_loss3(emd_dis, actual_dis, cfg):
    emd_dis = emd_dis  # bs x K
    actual_dis[actual_dis > 0.99] = 0.99
    # actual dis is in fact the matching aware loss, typically 0.2--0.4

    loss_part_2 = torch.sum(torch.abs(emd_dis - actual_dis), dim=1)

    return loss_part_2


def retrieval_regression_loss4(emd_dis, actual_dis):  # bs x k
    qij = F.softmax(emd_dis, dim=1)
    pij = F.softmax(actual_dis, dim=1)
    loss = torch.sum(torch.abs(qij - pij), dim=1)
    return loss


def retrieval_regression_loss5(emd_dis, actual_dis):  # K x bs
    # order according to actual dis
    _, idx_m = torch.sort(actual_dis, dim=0)  # idx_m  K x bs

    K = emd_dis.shape[0]
    bs = emd_dis.shape[-1]
    loss = 0.0
    loss_reg = 0.0
    # transpose for easy calculation
    emd_dis = emd_dis.permute(1, 0)
    _, order = torch.sort(idx_m, dim=0)  # order,  K x bs, rank of the matrix
    order = order.permute(1, 0)

    for i in range(K):
        if i == 0:
            loss += torch.abs(emd_dis[order == i] - emd_dis[order == i + 1]) * 3.0
        elif i == K - 1:
            loss += (emd_dis[order == i - 1] - emd_dis[order == i]) * 3.0
        else:
            left_loss = emd_dis[order == i - 1] - emd_dis[order == i]
            right_loss = emd_dis[order == i] - emd_dis[order == i + 1]
            left_loss[left_loss < 0.0] = 0.0
            right_loss[right_loss < 0.0] = 0.0
            loss += left_loss
            loss += right_loss
    loss_reg += emd_dis[order == 0]
    loss_reg += 1.0 - emd_dis[order == K - 1]
    return loss + 1.0 * loss_reg


def retrieval_regression_loss6(emd_dis, actual_dis):  # K x bs
    # order according to actual dis
    _, idx_m = torch.sort(actual_dis, dim=0)  # idx_m  K x bs

    K = emd_dis.shape[0]
    bs = emd_dis.shape[-1]
    loss_reg = 0.0
    # transpose for easy calculation
    emd_dis = emd_dis.permute(1, 0)
    _, order = torch.sort(idx_m, dim=0)  # order,  K x bs, rank of the matrix
    order = order.permute(1, 0)
    loss_reg += emd_dis[order == 0]
    loss_reg += 1.0 - emd_dis[order == K - 1]
    return loss_reg


def retrieval_regression_loss7(emd_dis, actual_dis):  # K x bs
    # order according to actual dis
    _, idx_m = torch.sort(actual_dis, dim=0)  # idx_m  K x bs

    K = emd_dis.shape[0]
    bs = emd_dis.shape[-1]
    loss = 0.0
    loss_reg = 0.0
    # transpose for easy calculation
    emd_dis = emd_dis.permute(1, 0)
    _, order = torch.sort(idx_m, dim=0)  # order,  K x bs, rank of the matrix
    order = order.permute(1, 0)

    loss += (emd_dis[order == 0] - emd_dis[order == 1]) * 3.0
    loss += (emd_dis[order == K - 2] - emd_dis[order == K - 1]) * 3.0

    return loss


def retrieval_regression_loss8(emd_dis, actual_dis):  # K x bs
    # order according to actual dis
    _, idx_m = torch.sort(actual_dis, dim=0)  # idx_m  K x bs

    K = emd_dis.shape[0]
    bs = emd_dis.shape[-1]
    loss = 0.0
    loss_reg = 0.0
    # transpose for easy calculation
    emd_dis = emd_dis.permute(1, 0)
    _, order = torch.sort(idx_m, dim=0)  # order,  K x bs, rank of the matrix
    order = order.permute(1, 0)

    for i in range(K):
        if i == 0:
            loss += torch.abs(emd_dis[order == i] - emd_dis[order == i + 1]) * 3.0
        elif i == K - 1:
            loss += (emd_dis[order == i - 1] - emd_dis[order == i]) * 3.0
        else:
            left_loss = emd_dis[order == 0] - emd_dis[order == i]
            right_loss = emd_dis[order == i] - emd_dis[order == K - 1]
            loss += left_loss
            loss += right_loss
    return loss


def retrieval_regression_loss9(emd_dis, actual_dis):
    qij = F.softmax(emd_dis, dim=0)  # K x bs
    pij = F.softmax(actual_dis, dim=0)
    loss = torch.abs(qij - pij)  # K x bs

    _, idx_m = torch.sort(actual_dis, dim=0)  # idx_m  K x bs

    K = emd_dis.shape[0]
    bs = emd_dis.shape[-1]
    loss_reg = 0.0
    # transpose for easy calculation
    emd_dis = emd_dis.permute(1, 0)
    _, order = torch.sort(idx_m, dim=0)  # order,  K x bs, rank of the matrix
    order = order.permute(1, 0)

    reg = emd_dis[order == K - 1] - emd_dis[order == 0]
    reg[reg < 0.0] = 0.0  # cut   size: bs

    return loss


# the result seems to ignore the target branch
# I guess using this loss can not handle the problem
def retrieval_classification_loss(emd_dis, actual_dis, cfg):
    # emd_dis : bs x feature dim x nofsource
    # actual dis : bs x nofsource
    emd_dis = emd_dis.permute(0, 2, 1)  # now bs x nofsource x result mat

    _, idx_m = torch.sort(actual_dis, dim=1)  # bs x k
    loss = 0.0

    _, order = torch.sort(idx_m, dim=1)  # order,  bs x k

    for i in range(cfg["batch_size"]):
        order_now = order[i, :]
        # order to one hot
        cls_loss = F.cross_entropy(emd_dis[i, :, :], order_now)
        loss += cls_loss

    return loss / cfg["batch_size"]


def retrieval_tuple_loss(order, actual_dis, cfg):
    _, idx_m = torch.sort(actual_dis, dim=1)  # bs x k
    _, order = torch.sort(idx_m, dim=1)  # order,  bs x k
    # order  bs x k
    positive_indices = idx_m[:, 0]
    negative_indices = idx_m[:, cfg["K"] - cfg["num_negative"]:]
    loss = 0.0
    for i in range(cfg["batch_size"]):
        positive_dis = order[i, positive_indices[i]]  # the best actual dis corresponds to best order
        negative_dis = order[i, negative_indices[i, :]]
        loss_now = positive_dis - negative_dis + cfg["margin"]
        loss_now = torch.max(loss_now, torch.zeros(loss_now.shape, device=cfg["device"]))
        loss += torch.mean(loss_now)
    return loss / cfg["batch_size"]


def retrieval_sort_rank_loss(emd_dis, actual_dis, cfg):
    # order according to actual dis
    _, idx_m = torch.sort(actual_dis, dim=1)  # idx_m  bs x k
    _, order = torch.sort(idx_m, dim=1)  # order,  bs x k
    K = cfg["K"]
    loss = 0.0
    for i in range(K - 1):
        for j in range(i + 1, K):
            loss_now = (emd_dis[order == i] - emd_dis[order == j] + 1.0) / (j - i)
            loss += loss_now
    return loss / cfg["batch_size"]


def retrieval_regression_loss_f(emd_dis, actual_dis, cfg):
    actual_dis[actual_dis > cfg["max_re"]] = cfg["max_re"]
    actual_dis[actual_dis < cfg["min_re"]] = cfg["min_re"]
    actual_dis = (actual_dis - cfg["min_re"]) / (cfg["max_re"] - cfg["min_re"])

    loss_part_2 = torch.sum(torch.abs(emd_dis - actual_dis), dim=1)

    return loss_part_2


def residual_retrieval_loss(x, x_source, residuals):
    # matching_m: typically bs x 1024 x 2048
    # x is the input point, bs x numpoints x 3, x_source is bs x 2048 x 3
    # find the nearest neighbor of x and x_source
    _, _, nn = ops.knn_points(x, x_source, K=1, return_nn=True)
    nn = nn.squeeze(2)  # bs x 1024 x 3
    res_nn = x + residuals - nn
    residual_loss = torch.mean(torch.sum(torch.abs(res_nn), dim=-1))
    residual_loss_reg = torch.mean(torch.sum(torch.abs(residuals), dim=-1))
    return residual_loss, residual_loss_reg



