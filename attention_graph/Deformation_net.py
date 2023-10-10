import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, models, transforms
from attention_graph.attention_gnn import GraphAttentionNet


# for each node, decoder the corresponding parameters

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
            x = self.bn1(x)
            x = F.relu(x)
        elif self.use_norm == 'use_ln':
            x = F.relu(self.ln1(self.fc1(x)))
        elif self.use_norm == 'use_in':
            x = F.relu(self.in1(self.fc1(x)))

        else:
            x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


# first all object parts, together with global feature, target feature,
# are connected with graph neural network, then deocode the parameters with MLP
class DeformNet_design1(nn.Module):
    def __init__(self, input_dim, num_stages, num_heads, part_latent_dim=32,
                 graph_dim=128, output_dim=6, use_offset=False, use_norm='use_bn'):
        super(DeformNet_design1, self).__init__()
        self.input_dim = input_dim
        self.num_stages = num_stages
        self.num_heads = num_heads
        self.use_offset = use_offset
        self.output_dim = output_dim
        self.graph_dim = graph_dim
        ## for part latent code embedding
        self.part_encoding = NodeDecoder(part_latent_dim, 128, self.graph_dim, use_norm='None')
        self.Node_decoder = NodeDecoder(self.input_dim, 256, self.output_dim, use_norm='None')
        self.graph_attention_net = GraphAttentionNet(self.num_stages,
                                                     self.graph_dim, self.num_heads, use_offset=use_offset)

    def forward(self, global_f, target_f, part_f):
        # node that for the batch, we process each element individually
        # global_f : 256
        # target_f : 256
        # part_f : 32 x nofobj
        global_f = global_f.view(1, -1, 1)
        target_f = target_f.view(1, -1, 1)
        global_node = torch.cat([global_f, target_f], dim=-1)

        part_node = part_f.permute(1, 0)
        # encoding to 128
        part_node = self.part_encoding(part_node)
        part_node = part_node.unsqueeze(0)  # bs, dim, numofpart
        part_node = part_node.permute(0, 2, 1)

        global_node_a, part_node_a = self.graph_attention_net(global_node, part_node)
        numofpart = part_f.shape[-1]
        global_node_r = torch.cat([global_node_a[:, :, 0],
                                   global_node_a[:, :, 1]], dim=1).view(1, -1, 1).repeat(1, 1, numofpart)
        full_f = torch.cat([global_node_r, part_node_a], dim=1)  # 1x dim x numofpart
        full_f = full_f.permute(2, 1, 0).squeeze(2)
        out_res = self.Node_decoder(full_f)  # numofpart x 6
        return out_res

# for this network, we first use MLP to decode the result, and then use graph to predict a residual for refinement
class DeformNet_design2(nn.Module):
    def __init__(self, input_dim, num_stages, num_heads, part_latent_dim=32,
                 graph_dim=128, output_dim=6, use_offset=False, use_norm='use_bn'):
        super(DeformNet_design2, self).__init__()
        self.input_dim = input_dim
        self.num_stages = num_stages
        self.num_heads = num_heads
        self.use_offset = use_offset
        self.output_dim = output_dim
        self.graph_dim = graph_dim
        ## for part latent code embedding
        self.part_encoding = NodeDecoder(part_latent_dim + 6, 128, self.graph_dim, use_norm=use_norm)
        self.Node_decoder = NodeDecoder(self.input_dim, 256, self.output_dim, use_norm=use_norm)
        self.Graph_decoder = NodeDecoder(self.graph_dim * 3, 256, self.output_dim, use_norm=use_norm)
        self.graph_attention_net = GraphAttentionNet(self.num_stages,
                                                     self.graph_dim, self.num_heads, use_offset=use_offset)

    def forward(self, global_f, target_f, part_f, train_stage='Stage1'):
        # node that for the batch, we process each element individually
        nofpart = part_f.shape[-1]
        global_f_r = global_f.view(-1, 1).repeat(1, nofpart)
        target_f_r = target_f.view(-1, 1).repeat(1, nofpart)
        full_f = torch.cat([global_f_r, target_f_r, part_f], dim=0)
        individual_param = self.Node_decoder(full_f.permute(1, 0))   # 12 x 6
        if train_stage == 'Stage1':
            return individual_param


        global_f = global_f.view(1, -1, 1)
        target_f = target_f.view(1, -1, 1)
        # should the global node be detached?
        global_node = torch.cat([global_f, target_f], dim=-1)
        # part_node = part_f.unsqueeze(0)  # bs, dim, numofpart
        # encoding to 128
        part_node = torch.cat([part_f, individual_param.permute(1, 0)], dim=0)  # 6 + 32
        part_node = self.part_encoding(part_node.unsqueeze(0).permute(0, 2, 1))
        part_node = part_node.permute(0, 2, 1)  # bs x
        global_node_a, part_node_a = self.graph_attention_net(global_node, part_node)
        global_node_r = torch.cat([global_node_a[:, :, 0],
                                   global_node_a[:, :, 1]], dim=1).view(1, -1, 1).repeat(1, 1, nofpart)
        full_f = torch.cat([global_node_r, part_node_a], dim=1)  # 1x dim x numofpart
        full_f = full_f.permute(2, 1, 0).squeeze(2)
        out_res = self.Graph_decoder(full_f)  # numofpart x 6
        return out_res + individual_param, out_res

if __name__ == '__main__':
    import numpy as np

    global_f = np.random.randn(256)
    target_f = np.random.randn(256)
    part_f = np.random.randn(32, 12)
    global_f = torch.from_numpy(global_f.astype(np.float32)).cuda()
    target_f = torch.from_numpy(target_f.astype(np.float32)).cuda()
    part_f = torch.from_numpy(part_f.astype(np.float32)).cuda()
    network = DeformNet_design1(256 * 3, 3, 4, graph_dim=256)
    network = network.cuda()
    our_res = network(global_f, target_f, part_f)
    s = 1
