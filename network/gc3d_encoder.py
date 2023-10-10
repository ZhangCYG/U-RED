# follow FS-Net
import torch.nn as nn
import network.P_3DGC as gcn3d
import torch
import torch.nn.functional as F

# global feature num : the channels of feature from rgb and depth
# grid_num : the volume resolution

class GCN3D_ENCODER(nn.Module):
    def __init__(self, gcn_n_num=10, gcn_sup_num=7):
        super(GCN3D_ENCODER, self).__init__()
        self.neighbor_num = gcn_n_num
        self.support_num = gcn_sup_num

        # 3D convolution for point cloud
        self.conv_0 = gcn3d.Conv_surface(kernel_num=128, support_num=self.support_num)
        self.conv_1 = gcn3d.Conv_layer(128, 128, support_num=self.support_num)
        self.pool_1 = gcn3d.Pool_layer(pooling_rate=4, neighbor_num=4)
        self.conv_2 = gcn3d.Conv_layer(128, 256, support_num=self.support_num)
        self.conv_3 = gcn3d.Conv_layer(256, 256, support_num=self.support_num)
        self.pool_2 = gcn3d.Pool_layer(pooling_rate=4, neighbor_num=4)
        self.conv_4 = gcn3d.Conv_layer(256, 256, support_num=self.support_num)

        self.dim_fuse = sum([128, 128, 256, 256, 256])

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)

        self.conv1d_block = nn.Sequential(
            nn.Conv1d(self.dim_fuse, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )


    def forward(self,
                vertices: "tensor (bs, vetice_num, 3)",
                ):
        """
        Return: (bs, vertice_num, class_num)
        """
        #  concate feature
        bs, vertice_num, _ = vertices.size()

        neighbor_index = gcn3d.get_neighbor_index(vertices, self.neighbor_num)
        # ss = time.time()
        fm_0 = F.relu(self.conv_0(neighbor_index, vertices), inplace=True)

        fm_1 = F.relu(self.bn1(self.conv_1(neighbor_index, vertices, fm_0).transpose(1, 2)).transpose(1, 2),
                      inplace=True)
        v_pool_1, fm_pool_1 = self.pool_1(vertices, fm_1)
        # neighbor_index = gcn3d.get_neighbor_index(v_pool_1, self.neighbor_num)
        neighbor_index = gcn3d.get_neighbor_index(v_pool_1,
                                                  min(self.neighbor_num, v_pool_1.shape[1] // 8))
        fm_2 = F.relu(self.bn2(self.conv_2(neighbor_index, v_pool_1, fm_pool_1).transpose(1, 2)).transpose(1, 2),
                      inplace=True)
        fm_3 = F.relu(self.bn3(self.conv_3(neighbor_index, v_pool_1, fm_2).transpose(1, 2)).transpose(1, 2),
                      inplace=True)
        v_pool_2, fm_pool_2 = self.pool_2(v_pool_1, fm_3)
        # neighbor_index = gcn3d.get_neighbor_index(v_pool_2, self.neighbor_num)
        neighbor_index = gcn3d.get_neighbor_index(v_pool_2, min(self.neighbor_num,
                                                                v_pool_2.shape[1] // 8))
        fm_4 = self.conv_4(neighbor_index, v_pool_2, fm_pool_2)
        f_global = fm_4.max(1)[0]  # (bs, f)

        nearest_pool_1 = gcn3d.get_nearest_index(vertices, v_pool_1)
        nearest_pool_2 = gcn3d.get_nearest_index(vertices, v_pool_2)
        fm_2 = gcn3d.indexing_neighbor(fm_2, nearest_pool_1).squeeze(2)
        fm_3 = gcn3d.indexing_neighbor(fm_3, nearest_pool_1).squeeze(2)
        fm_4 = gcn3d.indexing_neighbor(fm_4, nearest_pool_2).squeeze(2)
        feat = torch.cat([fm_0, fm_1, fm_2, fm_3, fm_4], dim=2)

        feat = self.conv1d_block(feat.permute(0, 2, 1))

        return f_global, feat
    #   f global bs x 512
    #   feat   bs x 256 x num_points

if __name__ == '__main__':
    import numpy as np

    points = np.random.random(size=(8, 2048, 3))
    points = torch.from_numpy(points.astype(np.float32)).cuda()
    network = GCN3D_ENCODER()
    network = network.cuda()
    f_global, feat = network(points)
    s = 1

