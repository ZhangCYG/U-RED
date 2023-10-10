import torch
import torch.nn as nn
import torch.nn.functional as F

from attention_graph.attention_gnn import GraphAttentionNet
from attention_graph.attention_utils import FeedForwardNet_norm


# for this net, the input
class NodeDecoder(nn.Module):
    def __init__(self, input_dim, intermediate_layer, embedding_size, use_norm='use_bn'):
        super(NodeDecoder, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, intermediate_layer)
        self.fc2 = nn.Linear(intermediate_layer, embedding_size)
        self.use_norm = use_norm
        if use_norm == 'use_bn':
            self.bn1 = nn.BatchNorm1d(intermediate_layer)
        if use_norm == 'use_ln':
            self.ln1 = nn.LayerNorm(intermediate_layer, elementwise_affine=True)
        if use_norm == 'use_in':
            self.in1 = nn.InstanceNorm1d(intermediate_layer)

    def forward(self, x):
        if self.use_norm == 'use_bn':
            x = self.fc1(x)
            x = x.permute(0, 2, 1)
            x = self.bn1(x)
            x = F.relu(x)
            x = x.permute(0, 2, 1)
        elif self.use_norm == 'use_ln':
            x = F.relu(self.ln1(self.fc1(x)))
        elif self.use_norm == 'use_in':
            x = F.relu(self.in1(self.fc1(x)))

        else:
            x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class DeformNet_MatchingNet(nn.Module):
    def __init__(self, input_dim, num_stages=2, num_heads=4, part_latent_dim=256,
                 graph_dim=128, output_dim=6, use_offset=False, point_f_dim=256,
                 points_num=2048, max_num_parts=12, matching=True):
        super(DeformNet_MatchingNet, self).__init__()
        self.input_dim = input_dim
        self.num_stages = num_stages
        self.num_heads = num_heads
        self.use_offset = use_offset
        self.output_dim = output_dim
        self.graph_dim = graph_dim
        self.point_f_dim = point_f_dim
        self.points_num = points_num
        self.max_num_parts = max_num_parts
        # network modules
        # 1 encode the part code into target dimension
        # for deformation
        self.part_encoding = FeedForwardNet_norm([part_latent_dim, 128, self.graph_dim], use_norm='None')
        self.param_decoder = FeedForwardNet_norm([self.input_dim, 256, self.output_dim], use_norm='None')
        # self.param_decoder = FeedForwardNet_norm([self.input_dim + 32, 256, self.output_dim], use_norm='None')  # add noise
        self.graph_attention_net = GraphAttentionNet(self.num_stages,
                                                     self.graph_dim, self.num_heads, use_offset=use_offset)

        # for matching
        self.matching = matching
        if matching:
            self.matching_net = FeedForwardNet_norm([self.point_f_dim + self.graph_dim * 2, 512,
                                                     1024, self.points_num], use_norm='use_bn')
        else:
            self.matching_net = None

    def forward(self, global_f, target_f, part_f, per_point_f):
        # node that for the batch, we process each element individually
        # global_f : bs x 256   full or partial global source feature
        # target_f : bs x 256
        # part_f : bs x {32 x nofobj} list
        # per_point_f: bs x 256 x 1024   # target occlusion per point, its matching matrix to source
        bs = global_f.shape[0]
        result_list = []
        for i in range(bs):
            global_f_now = global_f[i, :].view(1, -1, 1)
            target_f_now = target_f[i, :].view(1, -1, 1)
            global_node = torch.cat([global_f_now, target_f_now], dim=-1)

            part_node = part_f[i].unsqueeze(0)
            part_node = part_node.permute(0, 2, 1)
            curr_num_parts = part_node.shape[-1]
            # encoding to 128
            part_node = self.part_encoding(part_node)  # bs, dim, numofpart

            global_node_a, part_node_a = self.graph_attention_net(global_node, part_node)
            global_node_r = torch.cat([global_node_a[:, :, 0],
                                       global_node_a[:, :, 1]], dim=1).view(1, -1, 1).repeat(1, 1, curr_num_parts)
            full_f = torch.cat([global_node_r, part_node_a], dim=1)  # 1x dim x numofpart

            # # add noise
            # random_noise = torch.normal(mean=0., std=1., size=(1, 32, curr_num_parts)).to(full_f.device)
            # full_f = torch.cat([full_f, random_noise], dim=1)

            params = self.param_decoder(full_f)  # numofpart x 6
            params = params.squeeze().permute(1, 0)
            if curr_num_parts < self.max_num_parts:
                dummy_params = torch.zeros((self.max_num_parts - curr_num_parts, 6),
                                           dtype=torch.float, device=params.device)
                params = torch.cat([params, dummy_params], dim=0)
            # print(params)
            params = params.contiguous().view(-1, 1)
            result_list.append(params)
        if self.matching:
            global_f_per_point = global_f.view(bs, -1, 1).repeat(1, 1, per_point_f.shape[-1])  # bs x 256 x 1024
            target_f_per_point = target_f.view(bs, -1, 1).repeat(1, 1, per_point_f.shape[-1])
            per_point_matching_f = torch.cat([global_f_per_point, target_f_per_point, per_point_f],
                                             dim=1)  # bs x 512 x 1024

            matching_m = self.matching_net(per_point_matching_f)  # bs x 1024 x 2048
            matching_m = matching_m.permute(0, 2, 1)
           # matching_m = F.softmax(matching_m, dim=-1)
        else:
            matching_m = None

        return result_list, matching_m


class re_residual_net(nn.Module):
    def __init__(self, input_dim, output_dim=3):
        super(re_residual_net, self).__init__()
        self.input_dim = input_dim
        self.residual_net = FeedForwardNet_norm([self.input_dim, 256, 256, 32, output_dim], use_norm='use_bn')

    def forward(self, concat_feature):
        # concat feature: bs x num_points x feature dim
        assert self.input_dim == concat_feature.shape[-1]
        concat_feature = concat_feature.permute(0, 2, 1)
        residual_value = self.residual_net(concat_feature)
        return residual_value.permute(0, 2, 1)




if __name__ == '__main__':
    import numpy as np

    global_f = np.random.random(size=(8, 256))
    target_f = np.random.random(size=(8, 256))
    part_f = np.random.random(size=(8, 32, 12))
    per_point_f = np.random.random(size=(8, 256, 512))
    global_f = torch.from_numpy(global_f.astype(np.float32)).cuda()
    target_f = torch.from_numpy(target_f.astype(np.float32)).cuda()
    part_f = torch.from_numpy(part_f.astype(np.float32)).cuda()
    per_point_f = torch.from_numpy(per_point_f.astype(np.float32)).cuda()
    network = DeformNet_MatchingNet(256 * 3, 3, 4, graph_dim=256)
    network = network.cuda()
    result_list, matching_m = network(global_f, target_f, part_f, per_point_f)
    s = 1
