import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from network.VN.vn_layers import *
from network.VN.vn_dgcnn_util import get_graph_feature


class vn_encoder(nn.Module):
    def __init__(self, cfg, num_class=40, normal_channel=False):
        super(vn_encoder, self).__init__()
        self.args = cfg
        self.n_knn = cfg['n_knn']

        self.conv1 = VNLinearLeakyReLU(2, 64 // 3)
        self.conv2 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3)
        self.conv3 = VNLinearLeakyReLU(64 // 3 * 2, 128 // 3)
        self.conv4 = VNLinearLeakyReLU(128 // 3 * 2, 256 // 3)

        self.conv5 = VNLinearLeakyReLU(256 // 3 + 128 // 3 + 64 // 3 * 2, 1024 // 3, dim=4, share_nonlinearity=True)

        self.std_feature = VNStdFeature(1024 // 3 * 2, dim=4, normalize_frame=False)

        self.linear1 = nn.Linear((1024 // 3) * 12, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, cfg['target_latent_dim'])

        self.per_point_f = nn.Linear((1024 // 3) * 6, cfg['target_latent_dim'])

        # calculate per_point

        if cfg['pooling'] == 'max':
            self.pool1 = VNMaxPool(64 // 3)
            self.pool2 = VNMaxPool(64 // 3)
            self.pool3 = VNMaxPool(128 // 3)
            self.pool4 = VNMaxPool(256 // 3)
        elif cfg['pooling'] == 'mean':
            self.pool1 = mean_pool
            self.pool2 = mean_pool
            self.pool3 = mean_pool
            self.pool4 = mean_pool

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(1)
        x = get_graph_feature(x, k=self.n_knn)
        x = self.conv1(x)
        x1 = self.pool1(x)

        x = get_graph_feature(x1, k=self.n_knn)
        x = self.conv2(x)
        x2 = self.pool2(x)

        x = get_graph_feature(x2, k=self.n_knn)
        x = self.conv3(x)
        x3 = self.pool3(x)

        x = get_graph_feature(x3, k=self.n_knn)
        x = self.conv4(x)
        x4 = self.pool4(x)

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)

        num_points = x.size(-1)
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)
        x, trans = self.std_feature(x)
        x = x.view(batch_size, -1, num_points)

        point_feat = x.permute(0, 2, 1)
        point_feat = self.per_point_f(point_feat)

        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn1(self.linear1(x)), negative_slope=0.2)

        global_f = self.linear2(x)

        return global_f, point_feat.permute(0, 2, 1)



if __name__ == '__main__':
    import numpy as np
    import json

    config_path = '../../config/unused/config_debug.json'
    config = json.load(open(config_path, 'rb'))
    points = np.random.random(size=(8, 3, 1024))
    points = torch.from_numpy(points.astype(np.float32)).cuda()
    network = vn_encoder(cfg=config)
    network = network.cuda()
    out = network(points)
    s = 1
