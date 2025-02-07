import torch, math, utils
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv, GATv2Conv

class AudioCNN1D(nn.Module):
    def __init__(self, num_channels_input, num_channels_hidden,
                 first_kernel_size=80, other_kernel_size=2, stride=16, max_pool_size=4):
        super().__init__()
        self.conv1 = nn.Conv1d(num_channels_input, num_channels_hidden, kernel_size=first_kernel_size, stride=stride)
        self.bn1 = nn.BatchNorm1d(num_channels_hidden)
        self.pool1 = nn.MaxPool1d(max_pool_size)
        self.conv2 = nn.Conv1d(num_channels_hidden, num_channels_hidden, kernel_size=other_kernel_size)
        self.bn2 = nn.BatchNorm1d(num_channels_hidden)
        self.pool2 = nn.MaxPool1d(max_pool_size)
        self.conv3 = nn.Conv1d(num_channels_hidden, 2 * num_channels_hidden, kernel_size=other_kernel_size)
        self.bn3 = nn.BatchNorm1d(2 * num_channels_hidden)
        self.pool3 = nn.MaxPool1d(max_pool_size)
        self.conv4 = nn.Conv1d(2 * num_channels_hidden, 2 * num_channels_hidden, kernel_size=other_kernel_size)
        self.bn4 = nn.BatchNorm1d(2 * num_channels_hidden)
        self.pool4 = nn.MaxPool1d(max_pool_size)
        self.num_channels_output = num_channels_hidden * 2

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

        return features
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
class AudioCNN2D(nn.Module):
    def __init__(self, num_channels_input, num_channels_hidden, 
                 first_kernel_size=(16, 6), other_kernel_size=2, stride=4, avg_pool_size=2):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels_input, num_channels_hidden, kernel_size=first_kernel_size, stride=stride)
        self.bn1 = nn.BatchNorm2d(num_channels_hidden)
        self.pool1 = nn.AvgPool2d(avg_pool_size)
        self.conv2 = nn.Conv2d(num_channels_hidden, num_channels_hidden, kernel_size=other_kernel_size)
        self.bn2 = nn.BatchNorm2d(num_channels_hidden)
        self.pool2 = nn.AvgPool2d(avg_pool_size)
        self.conv3 = nn.Conv2d(num_channels_hidden, 2 * num_channels_hidden, kernel_size=other_kernel_size)
        self.bn3 = nn.BatchNorm2d(2 * num_channels_hidden)
        self.pool3 = nn.AvgPool2d(avg_pool_size)
        self.conv4 = nn.Conv2d(2 * num_channels_hidden, 2 * num_channels_hidden, kernel_size=other_kernel_size)
        self.bn4 = nn.BatchNorm2d(2 * num_channels_hidden)
        self.pool4 = nn.AvgPool2d(avg_pool_size)
        self.num_channels_output = 2 * num_channels_hidden

    def forward(self, x, global_pooling=True):
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
        features = self.pool4(x)
        if global_pooling:
            features = F.avg_pool2d(x, kernel_size=(x.shape[-2], x.shape[-1])).squeeze(-1).squeeze(-1)

        return features
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, final_state_only=True):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.final_state_only = final_state_only
    
    def forward(self, X):
        X = X.permute(0, 2, 1)
        out_features, _ = self.gru(X)
        if self.final_state_only:
            out_features = out_features[:, -1, :]

        return out_features
    
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, final_state_only=True):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.final_state_only = final_state_only
    
    def forward(self, X:torch.Tensor):
        X = X.permute(0, 2, 1)
        out_features, _ = self.lstm(X)
        if self.final_state_only:
            out_features = out_features[:, -1, :]

        return out_features 
    
class EnsembleGCN(nn.Module):
    def __init__(self, 
                 num_feature_inputs, feature_embed_sizes, 
                 final_embed_size, cross_class_conv1d_kernel_size, cross_class_conv1d_stride,
                 dropout_rate, num_classes_output, query_size, num_features=2):
        super().__init__()
        self.num_features = num_features
        self.dropout = nn.Dropout(p=dropout_rate)
        self.num_classes_output = num_classes_output
        self.query_size = query_size

        # 所有 特征压缩或放大部分的 GCNConv层
        self.feature_gconvs = nn.ModuleList(
            [GCNConv(num_input, embed_size) for num_input, embed_size in zip(num_feature_inputs, feature_embed_sizes)]
            )

        # 类内传播信息的 GCNConv层
        self.in_class_gconv = GCNConv(sum(feature_embed_sizes) + num_classes_output, final_embed_size)
        
        # 跨类传播信息的 GATConv层
        self.cross_class_gat = GATv2Conv(sum(feature_embed_sizes) + num_classes_output, final_embed_size)

        # 对经GATConv传播后的信息(类间分布)进行压缩的 Conv1d层
        self.cross_class_conv1d = nn.Sequential(
            nn.Unflatten(dim=1, unflattened_size=(1, -1)),
            nn.Conv1d(1, 1, kernel_size=cross_class_conv1d_kernel_size, stride=cross_class_conv1d_stride),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Sigmoid()
        )

        # 最后的进行分类预测的 GCNConv层
        fc_size = final_embed_size + math.floor((final_embed_size - cross_class_conv1d_kernel_size) / cross_class_conv1d_stride) + 1
        self.label_gconv = GCNConv(fc_size, num_classes_output)

    def to_one_hot(self, labels: torch.Tensor):
        num_samples = labels.shape[0]
        one_hot = torch.zeros((num_samples, self.num_classes_output), device=labels.device)
        one_hot[torch.arange(num_samples - self.num_classes_output * self.query_size), labels[:num_samples - self.num_classes_output * self.query_size]] = 1
        one_hot[-self.num_classes_output * self.query_size:, :] = 1 / self.num_classes_output

        return one_hot

    def forward(self, features_list: list, labels: torch.Tensor):
        assert len(features_list) == self.num_features, f"应有 {self.num_features} 种特征, 但输入了 {len(features_list)}种"

        one_hot = self.to_one_hot(labels)
        # 为QuerySet修改标签内容, 将所有QuerySet样本的标签, 按类置为不同的负数
        temp_labels = torch.clone(labels)
        temp_labels[-self.num_classes_output * self.query_size:] = torch.arange(-1, -(self.num_classes_output * self.query_size + 1), -1, dtype=temp_labels.dtype)
        
        # 类内和类间邻接矩阵
        in_class_edge_index = utils.in_class_adj_matrix(temp_labels)
        cross_class_edge_index = utils.cross_class_adj_matrix(temp_labels)

        # 对每种提取的特征应用对应的GCNConv层进行压缩或放大
        feature_embeddings = [self.dropout(F.leaky_relu(gconv(features, in_class_edge_index))) 
                              for gconv, features in zip(self.feature_gconvs, features_list)]
        
        # 连接压缩或放大后的特征
        cated_features = torch.concat(feature_embeddings + [one_hot], dim=-1) 
        gcn_final_embed = F.leaky_relu(self.in_class_gconv(cated_features, in_class_edge_index))
        gat_final_embed = F.elu(self.cross_class_gat(cated_features, cross_class_edge_index))

        # Conv1d压缩GATConv后的类分布特征
        gat_final_embed = self.cross_class_conv1d(gat_final_embed)

        # 连接类内特征和类间分布特征
        fc_embed = torch.concat([gcn_final_embed, gat_final_embed], dim=-1)

        # 预测类别
        final_edge_index = utils.final_pooling_adj_matrix(labels, self.num_classes_output * self.query_size)
        output = self.label_gconv(fc_embed, final_edge_index)

        return output

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
class EnsembleNet(nn.Module):
    def __init__(self, cnn1d:nn.Module, ensemble_gcn:nn.Module, in_data_type:str,
                 cnn2d:nn.Module=None, lstm:nn.Module=None, dropout_rate:float=0.5):
        """
        Args:\n
            cnn1d : 一个nn.Module, 是一个1D的卷积神经网络
            cnn2d : 一个nn.Module, 是一个2D的卷积神经网络
            lstm : 一个nn.Module, 是一个基于LSTM的神经网络
            in_data_type : 指示输入数据类型的变量, 为 'seq' 或 'audio'
            dropout_rate : nn.Dropout中的 p值
        集成特征提取器和图卷积网络的总网络
        """
        assert in_data_type in {'seq', 'audio'}, "in_data_type必须为'seq'或'audio'"
        super().__init__()
        self.cnn1d = cnn1d
        self.cnn2d = cnn2d
        self.lstm = lstm
        self.ensemble_gcn = ensemble_gcn
        self.in_data_type = in_data_type
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, waveforms, spectrograms=None, labels=None):
        cnn1d_features = self.dropout(self.cnn1d(waveforms))
        if self.in_data_type == 'seq' and self.lstm is not None:
            lstm_features = self.dropout(self.lstm(waveforms))
            out = self.ensemble_gcn([cnn1d_features, lstm_features], labels)
        elif self.in_data_type == 'audio' and self.cnn2d is not None:
            cnn2d_features = self.dropout(self.cnn2d(spectrograms))
            out = self.ensemble_gcn([cnn1d_features, cnn2d_features], labels)

        return out