import torch, utils
from torch import nn as nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv

class AudioCNN(nn.Module):
    def __init__(self, embed_size, num_channels_input=1, num_channels_hidden=32, num_classes_output=32, stride=16):
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
        self.num_channels_output = num_channels_hidden * 2
        self.feature_extract_fc = nn.Linear(2 * num_channels_hidden, embed_size)
        self.classify_fc = nn.Linear(2 * num_channels_hidden, num_classes_output)
        self.embed_size = embed_size

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
        features = F.avg_pool1d(x, x.shape[-1]).squeeze(-1)
        if self.training:
            y_hat = self.classify_fc(features)
            return y_hat
        else:
            return features
        #features = self.feature_extract_fc(x)
    #x = x.permute(0, 2, 1)
    #,F.log_softmax(x, dim=2)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
class GCN(nn.Module):
    def __init__(self, num_features_inputs, num_classes_output):
        super().__init__()
        self.gcn = GCNConv(num_features_inputs, 16)
        self.out = torch.nn.Linear(16, num_classes_output)

    def forward(self, x, edge_index, edge_weight):
        embeddings = self.gcn(x, edge_index, edge_weight=edge_weight).relu()
        out = self.out(embeddings)
        return out
    
class EnsembleNet(nn.Module):
    def __init__(self, cnn_embed_size, cnn_num_channels_input=1, cnn_num_channels_hidden=32, cnn_stride=16,
                num_classes_outputs=32):
        super().__init__()
        self.cnn = AudioCNN(embed_size=cnn_embed_size, num_channels_input=cnn_num_channels_input,
                            num_channels_hidden=cnn_num_channels_hidden, stride=cnn_stride)
        self.gcn = GCN(num_features_inputs=self.cnn.embed_size,
                       num_classes_output=num_classes_outputs)

    def forward(self, X, y):
        encoded_X = self.cnn(X)
        graph = utils.graph_construction(encoded_X, y)
        pred_y = self.gcn(graph.x, graph.edge_index, edge_weight=graph.edge_weight)
        return pred_y