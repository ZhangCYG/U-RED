import torch
import torch.nn.functional as F

def compute_pc_consistency(pc1, pc2):
    res_pc = pc1 - pc2       # bs x num x 3
    res = torch.sum(res_pc * res_pc, dim=-1)
    loss = torch.mean(res)
    return loss

def compute_matching_m_consistency(m1, m2, m_list):
    # m1, from full
    # m2 from partial
    # m_list: the corresponding matrix of m1 to m2
    bs = m1.shape[0]
    eu_loss = 0.0
    for i in range(bs):
        m_2_from_1 = m1[i, m_list[i], :]   # 1024 x 2048
        eu_loss += (F.kl_div(torch.log(m2[i, :, :] + 1e-6), m_2_from_1, reduction='batchmean')
                    + F.kl_div(torch.log(m_2_from_1 + 1e-6), m2[i, :, :], reduction='batchmean'))
    return eu_loss / bs

def compute_param_consistency(param1, param2):
    res_param = param1 - param2
    res = torch.sqrt(torch.sum(res_param * res_param, dim=-1))
    loss = torch.mean(res)
    return loss



