import torch
import torch.nn as nn
import torch.nn.functional as F


class STN3D(nn.Module):
    def __init__(self, input_channels=3):
        super(STN3D, self).__init__()
        self.input_channels = input_channels
        self.mlp1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, input_channels * input_channels),
        )

    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)
        x = self.mlp1(x)
        x = F.max_pool1d(x, num_points).squeeze(2)
        x = self.mlp2(x)
        I = torch.eye(self.input_channels).view(-1).to(x.device)
        x = x + I
        x = x.view(-1, self.input_channels, self.input_channels)
        return x


class TargetEncoder(nn.Module):
    def __init__(self, embedding_size=256, input_channels=3):
        super(TargetEncoder, self).__init__()
        self.input_channels = input_channels
        self.stn1 = STN3D(input_channels)
        self.stn2 = STN3D(64)
        self.mlp1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.per_point_out = nn.Sequential(
            nn.Conv1d(1024, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, 1)
        )
        self.fc = nn.Linear(1024, embedding_size)

    def forward(self, x):
        batch_size = x.shape[0]
        num_points = x.shape[1]
        x = x[:, :, : self.input_channels]
        x = x.transpose(2, 1)  # transpose to apply 1D convolution
        x = self.mlp1(x)
        x = self.mlp2(x)
        per_point = self.per_point_out(x)
        x = F.max_pool1d(x, num_points).squeeze(2)  # max pooling
        x = self.fc(x)
        return x, per_point


class SrcEncoder(nn.Module):
    def __init__(self, embedding_size=256, input_channels=3):
        super(SrcEncoder, self).__init__()
        self.input_channels = input_channels
        self.stn1 = STN3D(input_channels)
        self.stn2 = STN3D(64)
        self.mlp1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.fc = nn.Linear(1024, embedding_size)

    def forward(self, x):
        batch_size = x.shape[0]
        num_points = x.shape[1]
        x = x[:, :, : self.input_channels]
        x = x.transpose(2, 1)  # transpose to apply 1D convolution
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = F.max_pool1d(x, num_points).squeeze(2)  # max pooling
        x = self.fc(x)
        return x
