import torch.nn as nn
import torch.optim as optim
import torch_geometric.utils
import torch, random, torchaudio, os, time, math, torch_geometric
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from torch_geometric.nn import GCNConv
import networkx as nx
from torchvision import transforms as transforms
import torchaudio.transforms as audio_transforms
import nets

class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None
        self.elapsed_time_sum = 0

    def start(self):
        self.start_time = time.time()
        self.end_time = None
        self.elapsed_time = None

    def stop(self):
        if self.start_time is None:
            raise ValueError("Timer has not been started.")
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        self.elapsed_time_sum += self.elapsed_time

    def get_elapsed_time(self):
        if self.elapsed_time is None:
            raise ValueError("Timer has not been stopped yet.")
        return self.elapsed_time

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
        # print(f"Elapsed time: {self.get_elapsed_time():.4f} seconds")

class Accumulator:
    """累加多个变量的实用程序类"""
    def __init__(self, n):
        self.data = [0.0]*n

    def add(self, *args):
        """在data的对应位置加上对应的数"""
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class ResVisualization:
    def __init__(self, legend_name: tuple, num_epochs) -> None:
        self.res_dict = {name: [] for name in legend_name}
        self.num_epochs = num_epochs

    def plot_res(self):
        for legend_name, data in self.res_dict.items():
            plt.plot(list(range(self.num_epochs)), data, label=legend_name)
        plt.title("Result")
        plt.xlabel("num_epochs")
        plt.ylabel("ResultValue")
        plt.legend()
        plt.show()

class Args:
    def __init__(self, seed):
        self.device = try_gpu()
        self.seed = seed

def try_gpu(i=0):
    """如果存在,返回gpu(i), 否则返回cpu"""
    if torch.cuda.device_count() >= i+1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')
        
def graph_construction(feature_vecs, labels):
    """返回一个torch_geometric.data.Data类, 代表一张图"""
    num_nodes = feature_vecs.shape[0]
    adj_matrix = torch.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                continue
            adj_matrix[i][j] = 1 / torch.sum(torch.pow(feature_vecs[i] - feature_vecs[j], 2), 0)
    edge_index, edge_weight = torch_geometric.utils.dense_to_sparse(adj_matrix)
    return Data(x=feature_vecs, edge_index=edge_index, edge_weight=edge_weight, y=labels)

def graph_visualize(graph_dataset):
    graph = to_networkx(graph_dataset, to_undirected=True)
    plt.figure(figsize=(12,12))
    plt.axis('off')
    nx.draw_networkx(graph,
                    pos=nx.spring_layout(graph, seed=0),
                    with_labels=True,
                    node_size=800,
                    node_color=graph_dataset.y.cpu().numpy(),
                    cmap="hsv",
                    vmin=-2,
                    vmax=3,
                    width=0.8,
                    edge_color="grey",
                    font_size=14
                    )
    
def graph_conv_train(gcn:nn.Module, cnn, train_loader, num_epochs, lr, device):

    def init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
    
        elif isinstance(module, GCNConv):
            nn.init.xavier_uniform_(module.lin.weight)
    
    def get_num_correct_query_pred(pred_y, y, support_size):
        return torch.sum(pred_y[support_size:].argmax(dim=1) == y[support_size:])

    gcn.apply(init_weights)
    gcn.to(device).train()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(gcn.parameters(), lr=lr)
    losses, accs = [], []

    for epoch in range(1, num_epochs+1):
        metric = Accumulator(4) # 一个epoch经过样本数 一个epoch所有样本损失和 一个epoch的query总数 一个epoch正确分类的query总数
        for support_set, query_set in train_loader:
            waveforms, labels = [], []
            support_size, query_size = len(support_set), len(query_set)

            for waveform, label in support_set:
                waveforms.append(waveform.squeeze(0)),
                labels.append(label)

            for waveform, label in query_set:
                waveforms.append(waveform.squeeze(0)),
                labels.append(label)

            waveforms = torch.stack(waveforms).to(device)
            labels = torch.tensor(labels, device=device)
            feature_vecs = feature_extraction(cnn, waveforms)
            graph = graph_construction(feature_vecs, labels)
            x, y, edge_index, edge_weight = graph.x, graph.y, graph.edge_index, graph.edge_weight
            edge_index = edge_index.to(device)
            edge_weight = edge_weight.to(device)

            optimizer.zero_grad()
            pred_lables = gcn(x, edge_index, edge_weight)
            loss = loss_function(pred_lables, y)
            loss.backward()
            optimizer.step()
            metric.add(feature_vecs.shape[0], loss.item(),
                       query_size, get_num_correct_query_pred(pred_lables, y, support_size))
            print(epoch, metric[2])
            
        accs.append(metric[3] / metric[2])
        losses.append(metric[1] / metric[0])
        if epoch % 50 == 0 :
            print(f"Epoch:{epoch}  |  Loss :{losses[-1]:.3f}  |  Acc: {accs[-1]*100:.2f}%")

    plt.plot(list(range(num_epochs)), losses, label='Loss')
    plt.plot(list(range(num_epochs)), accs, label='Acc')
    plt.legend()
    
def cnn_train(model, num_epochs, train_loader, device):

    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    loss_function = nn.CrossEntropyLoss()
    model.to(device).train()
    losses = []

    for epoch in range(1, num_epochs+1):
        metric = Accumulator(2) # 一个epoch的损失 这个epoch经过的batch数
        for support_set, _ in train_loader:
            waveforms, labels = [], []
            for waveform, label in support_set:
                waveforms.append(waveform.squeeze(0)),
                labels.append(label)

            waveforms = torch.stack(waveforms).to(device)
            labels = torch.tensor(labels, device=device)

            optimizer.zero_grad()
            _, outputs = model(waveforms)
            outputs = outputs.squeeze(1)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            metric.add(loss.item(), 1)

        #if epoch%10 == 0:
        print(f"Train Epoch: {epoch}   |   Loss: {metric[0] / metric[1]:.3f}")
        losses.append(metric[0])

    plt.plot(list(range(num_epochs)), losses)
    plt.title("Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

def feature_extraction(cnn, waveforms):
    cnn.eval()
    feature_vecs = cnn(waveforms)
    feature_vecs = feature_vecs.cpu()
    return feature_vecs

def train(net:nets.EnsembleNet, train_loader, num_epochs, lr, device):

    def init_weights(module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d):
            nn.init.xavier_uniform_(module.weight)
        elif isinstance(module, GCNConv):
            nn.init.xavier_uniform_(module.lin.weight)

    def get_num_correct_query_pred(pred_y, y, support_size):
        return torch.sum(pred_y[support_size:].argmax(dim=1) == y[support_size:])

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    loss_function = nn.CrossEntropyLoss()
    net.to(device).train()
    net.apply(init_weights)
    losses = []
    accs = []

    for epoch in range(1, num_epochs+1):
        metric = Accumulator(4) # 一个epoch经过样本数 一个epoch所有样本损失总量 一个epoch的query总数 一个epoch正确分类的query总数
        for support_set, query_set in train_loader:
            support_size, query_size = len(support_set), len(query_set)
            waveforms, labels = [], []
            for waveform, label in support_set:
                waveforms.append(waveform.squeeze(0)),
                labels.append(label)
            waveforms = torch.stack(waveforms).to(device)
            labels = torch.tensor(labels, device=device)

            pred_labels = net(waveforms, labels)
            loss = loss_function(pred_labels, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            metric.add(waveforms.shape[0], loss.item(),
                        query_size, get_num_correct_query_pred(pred_labels, labels, support_size))

        accs.append(metric[3] / metric[2])
        losses.append(metric[1] / metric[0])
        if epoch % 10 == 0 :
            print(f"Epoch:{epoch}  |  Loss :{losses[-1]:.3f}  |  Acc: {accs[-1]*100:.2f}%")

    plt.plot(list(range(num_epochs)), losses, label='Loss')
    plt.plot(list(range(num_epochs)), accs, label='Acc')
    plt.title("Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss & Acc')

def to_graph_dataset(cnn:nn.Module, dataloader:torch.utils.data.DataLoader, device):
    """
    Device指的是CNN的device, 而不是最终形成的graph的device\n
    最终的graph会被放在cpu上
    """
    graphs = []
    for support_set, query_set in dataloader:
        waveforms, labels = [], []
        for waveform, label in support_set:
            waveforms.append(waveform.squeeze(0)),
            labels.append(label)
        for waveform, label in query_set:
            waveforms.append(waveform.squeeze(0)),
            labels.append(label)
        waveforms = torch.stack(waveforms).to(device)
        labels = torch.tensor(labels)

        feature_vecs = feature_extraction(cnn, waveforms)
        graph = graph_construction(feature_vecs, labels)
        graphs.append(graph)
    
    return graphs