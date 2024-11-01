import torch
from torch import nn as nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv

class AudioCNN(nn.Module):
    def __init__(self, num_channels_input=1, num_outputs=35, num_channels_hidden=32, stride=16):
        super().__init__()
        self.conv1 = nn.Conv1d(num_channels_input, num_channels_hidden, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(num_channels_hidden)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(num_channels_hidden, num_channels_hidden, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(num_channels_hidden)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(num_channels_hidden, 2 * num_channels_hidden, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * num_channels_hidden)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * num_channels_hidden, 2 * num_channels_hidden, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * num_channels_hidden)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * num_channels_hidden, num_outputs)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        features = x
        x = self.fc1(x)
        return features ,F.log_softmax(x, dim=2)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
class GCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.gcn = GCNConv(64, 16)
        self.out = torch.nn.Linear(16, 32)

    def forward(self, x, edge_index, edge_weight):
        embeddings = self.gcn(x, edge_index, edge_weight=edge_weight).relu()
        out = self.out(embeddings)
        return out