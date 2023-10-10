import torch
import torch.nn as nn
import torch.nn.functional as F
from network.simple_encoder import SrcEncoder as simple_encoder
from network.VN.vn_retrieval import vn_retrieval
from network.VN.vn_encoder import vn_encoder
from network.simple_encoder import SrcEncoder as simple_retrieval

class re_target_net(nn.Module):
    def __init__(self, cfg):
        super(re_target_net, self).__init__()
        self.cfg = cfg
        self.re_target_encoder = vn_retrieval(cfg)

    def forward(self, target_x):
        target_latent_codes = self.re_target_encoder(target_x.permute(0, 2, 1))  # bs x 256
        return target_latent_codes


class re_src_net(nn.Module):
    def __init__(self, cfg):
        super(re_src_net, self).__init__()
        self.cfg = cfg
        self.re_src_encoder = simple_encoder()
    def forward(self, src_x):
        src_latent_codes = self.re_src_encoder(src_x.permute(0, 2, 1))  # (bs xx k) x 256
        return src_latent_codes

class re_order_net(nn.Module):
    def __init__(self, cfg):
        super(re_order_net, self).__init__()
        self.cfg = cfg
        self.ordernet = nn.Sequential(
            nn.Conv1d(self.cfg["source_latent_dim"] * 2 + self.cfg["target_latent_dim"], 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 1, 1),  # bs x 1 x K
        )
    def forward(self, concat_latent_codes):
        order = self.ordernet(concat_latent_codes.permute(0, 2, 1))  # K x k x bs
        return order


class re_network(nn.Module):
    def __init__(self, cfg):
        super(re_network, self).__init__()
        self.cfg = cfg
        self.re_target_net = vn_retrieval(self.cfg)
        self.re_src_net = simple_encoder()
        self.ordernet = nn.Sequential(
            nn.Conv1d(self.cfg["source_latent_dim"] + self.cfg["target_latent_dim"], 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 1, 1),    # bs x 1
        )

    def forward(self, target_x, src_x):
        # target_x  bs x 1024 x 3
        # src_x  (bs x k) x 1024 x 3
        target_latent_codes = self.re_target_net(target_x.permute(0, 2, 1))  # bs x 256
        src_latent_codes = self.re_src_net(src_x.permute(0, 2, 1))   # (bs xx k) x 256
        src_latent_codes = src_latent_codes.view(self.cfg["K"], self.cfg["batch_size"], self.cfg["source_latent_dim"])
        target_latent_codes = target_latent_codes.unsqueeze(0).repeat(self.cfg["K"], 1, 1)
        concat_latent_codes = torch.cat([src_latent_codes, target_latent_codes], dim=-1)
        order = self.ordernet(concat_latent_codes.permute(0, 2, 1))   # K x bs
        order = torch.sigmoid(order)  # K x bs
        return order.squeeze(1)


