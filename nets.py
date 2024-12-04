import torch
from torch import nn as nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse

class AudioCNN(nn.Module):
    def __init__(self, num_channels_input, num_channels_hidden, num_classes_output, stride=16):
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
        self.classify_fc = nn.Linear(2 * num_channels_hidden, num_classes_output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.leaky_relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.leaky_relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.leaky_relu(self.bn4(x))
        x = self.pool4(x)
        features = F.avg_pool1d(x, x.shape[-1]).squeeze(-1)
        if self.training:
            y_hat = self.classify_fc(features)
            return y_hat
        else:
            #features = self.feature_extract_fc(x)
            return features
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
class AudioCNN2D(nn.Module):
    def __init__(self, num_channels_input, num_channels_hidden, num_classes_output, stride=4):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels_input, num_channels_hidden, kernel_size=(16, 6), stride=stride)
        self.bn1 = nn.BatchNorm2d(num_channels_hidden)
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(num_channels_hidden, num_channels_hidden, kernel_size=2)
        self.bn2 = nn.BatchNorm2d(num_channels_hidden)
        self.pool2 = nn.AvgPool2d(2)
        self.conv3 = nn.Conv2d(num_channels_hidden, 2 * num_channels_hidden, kernel_size=2)
        self.bn3 = nn.BatchNorm2d(2 * num_channels_hidden)
        self.pool3 = nn.AvgPool2d(2)
        self.conv4 = nn.Conv2d(2 * num_channels_hidden, 2 * num_channels_hidden, kernel_size=2)
        self.bn4 = nn.BatchNorm2d(2 * num_channels_hidden)
        self.pool4 = nn.AvgPool2d(2)
        self.num_channels_output = 2 * num_channels_hidden
        self.classify_fc = nn.Linear(self.num_channels_output, num_classes_output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.leaky_relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.leaky_relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.leaky_relu(self.bn4(x))
        x = self.pool4(x)
        features = F.avg_pool2d(x, kernel_size=((x.shape[-2], x.shape[-1]))).squeeze(-1).squeeze(-1)
        if self.training:
            y_hat = self.classify_fc(features)
            return y_hat
        else:
            return features
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)   
    

class EnsembleGCN(nn.Module):
    """
    GCN架构:
    将time_features送入一个time_GCN, 将freq_features送入一个frea_GCN,
    得到图卷积后的两种特征形状为 [n_samples, time_gcn_embed_sz] 和 [n_samples, freq_gcn_embed_sz]
    连接两种特征 -> features.shape = [n_samples ,time_gcn_embed_sz+freq_gcn_embed_sz]
    连接后的特征送入一个GCN, 再次卷积 -> [n_samples, gcn_embed_sz] -> [Linear] -> n_classes
    """
    def __init__(self, num_time_features_inputs, num_freq_features_inputs, 
                 time_embed_size, freq_embed_size, embed_size, num_classes_output):
        super().__init__()
        self.time_gconv = GCNConv(num_time_features_inputs, time_embed_size)
        self.freq_gconv = GCNConv(num_freq_features_inputs, freq_embed_size)
        self.concat_gconv = GCNConv(time_embed_size+freq_embed_size+num_classes_output, embed_size)
        self.out = nn.Linear(embed_size, num_classes_output)

    def to_one_hot(self, labels, num_classes, query_size):
        num_samples = labels.shape[0]
        one_hot = torch.zeros((num_samples, num_classes), device=labels.device)
        one_hot[:num_samples-query_size, labels] = 1

        return one_hot

    def graph_construction(self, feature_vecs, num_classes):
        """内嵌于EnsembleGCN的更新连接权重的函数"""
        num_nodes, vec_len = feature_vecs.shape[0], feature_vecs.shape[1]
        adj_matrix = torch.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    continue
                adj_matrix[i][j] = 1 / (torch.sum(torch.sqrt(torch.pow(feature_vecs[i, :vec_len-num_classes] - feature_vecs[j, :vec_len-num_classes], 2)), 0)+torch.tensor(1e-5))
        _, edge_weight = dense_to_sparse(adj_matrix)
        return edge_weight

    def forward(self, time_features, edge_index, time_edge_weight,
                freq_features, freq_edge_weight, labels, num_classes, query_size):
        time_embedding = F.leaky_relu(self.time_gconv(time_features, edge_index, edge_weight=time_edge_weight))
        freq_embedding = F.leaky_relu(self.freq_gconv(freq_features, edge_index, edge_weight=freq_edge_weight))
        concated_features = torch.concat([time_embedding, freq_embedding, self.to_one_hot(labels, num_classes, query_size)], dim=-1)
        with torch.no_grad():
            concated_edge_weight = self.graph_construction(concated_features, num_classes).to(time_embedding.device)
        embedding = F.leaky_relu(self.concat_gconv(concated_features, edge_index, edge_weight=concated_edge_weight))
        out = self.out(embedding)
        return out
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)