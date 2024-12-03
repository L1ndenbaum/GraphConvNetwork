import torch.nn as nn
import torch.optim as optim
import torch_geometric.utils
import torch, random, torchaudio, os, time, math, torch_geometric
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from torch_geometric.nn import GCNConv
import networkx as nx
import datasets, nets
from torch_geometric.loader import DataLoader as GeometricDataLoader

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
    def __init__(self, seed, N, K, query_size, SNR, transform):
        self.device = try_gpu()
        self.seed = seed
        self.SNR = SNR
        self.N = N
        self.K = K
        self.query_size = query_size
        if transform is None:
            self.transform = 'raw'
        else:
            self.transform = transform


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
    
def gcn_train(gcn, graph_train_loader, query_size, num_epochs, lr, args:Args):

    def init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0, std=1. / (10 ** 0.5))
        elif isinstance(module, torch_geometric.nn.GCNConv):
            nn.init.normal_(module.lin.weight, mean=0, std=1. / (10 ** 0.5))

    def get_num_correct_query_pred(pred_y, y, query_size):
        return torch.sum(pred_y[-query_size:].argmax(dim=1) == y[-query_size:])

    optimizer = optim.Adam(gcn.parameters(), lr=lr)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)
    loss_function = nn.CrossEntropyLoss()
    gcn.to(args.device).train()
    gcn.apply(init_weights)
    num_subplot_rows = len(graph_train_loader) if len(graph_train_loader) <= 4 else 4
    fig, axes = plt.subplots(num_subplot_rows, 2, squeeze=False)
    for graph_idx, graph in enumerate(graph_train_loader):
        losses, accs = [], []
        x, y = graph.x.to(args.device), graph.y.to(args.device)
        edge_index, edge_weight = graph.edge_index.to(args.device), graph.edge_weight.to(args.device)
        for epoch in range(1, num_epochs+1):
            metric = Accumulator(4) # 一个epoch经过样本数 gcn一个epoch所有样本损失总量 一个epoch的query总数 一个epoch正确分类的query总数
            optimizer.zero_grad()
            y_hat = gcn(x, edge_index, edge_weight)
            loss = loss_function(y_hat, y)
            loss.backward(retain_graph=True)
            optimizer.step()
            #scheduler.step()
            with torch.no_grad():
                metric.add(x.shape[0], loss.item(),
                        query_size, get_num_correct_query_pred(y_hat, y, query_size))
            if epoch % 50==0:
                print(f"在第{graph_idx+1}张图  |  Epoch:{epoch}  |  GCN_Loss :{metric[1]:.3f}  |  Acc:{metric[3]/metric[2]*100:.2f}%")
            losses.append(metric[1])
            accs.append(metric[3]/metric[2])
        del x, y, edge_index, edge_weight
        torch.cuda.empty_cache()

        if graph_idx+1 <= num_subplot_rows:
            axes[graph_idx, 0].plot(list(range(num_epochs)), losses)
            axes[graph_idx, 0].set_title('GCN Loss')
            axes[graph_idx, 1].plot(list(range(num_epochs)), accs)
            axes[graph_idx, 1].set_title('Query Accurancy')

    plt.tight_layout()
    res_folder = f"./Results/{args.N}-way-{args.K}-shots"
    if not os.path.exists(res_folder):
        os.makedirs(res_folder)
    plt.savefig(os.path.join(res_folder, f"{args.SNR}SNR-gcn-train-res-{args.N}-way-{args.K}-shots-"+args.transform))
    
def cnn_train(cnn, train_loader, num_epochs, lr, args:Args):

    def init_weights(module):
        if isinstance(module, nn.Linear) or isinstance(module, (nn.Conv1d, nn.Conv2d)):
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')

    optimizer = optim.Adam(cnn.parameters(), lr=lr, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=16, gamma=0.5)
    loss_function = nn.CrossEntropyLoss()

    cnn.to(args.device).train()
    cnn.apply(init_weights)
    losses = []

    for epoch in range(1, num_epochs+1):
        metric = Accumulator(2) # 一个epoch经过样本数 cnn一个epoch所有样本损失总量
        for support_set, query_set in train_loader:
            waveforms, labels = [], []
            for waveform, label in support_set:
                waveforms.append(waveform.squeeze(0)),
                labels.append(label)
            for waveform, label in query_set:
                waveforms.append(waveform.squeeze(0)),
                labels.append(label)
            waveforms = torch.stack(waveforms).to(args.device)
            labels = torch.tensor(labels, device=args.device)
            
            optimizer.zero_grad()
            y_hat = cnn(waveforms)
            loss = loss_function(y_hat, labels)
            loss.backward()
            optimizer.step()
            #scheduler.step()
            metric.add(waveforms.shape[0], loss.item())
            del waveforms, labels
            torch.cuda.empty_cache()
        losses.append(metric[1])

        print(f"Epoch:{epoch}  |  CNN_Loss :{metric[1]:.3f}")
    fig = plt.figure()
    plt.plot(list(range(num_epochs)), losses, label='Loss')
    plt.title('CNN Loss')
    res_folder = f"./Results/{args.N}-way-{args.K}-shots"
    if not os.path.exists(res_folder):
        os.makedirs(res_folder)
    plt.savefig(os.path.join(res_folder, f"{args.SNR}SNR-cnn-train-res-{args.N}-way-{args.K}-shots-"+args.transform))

def feature_extraction(cnn, waveforms):
    cnn.eval()
    cnn.to(waveforms.device)
    with torch.no_grad():
        feature_vecs = cnn(waveforms)
    return feature_vecs.cpu()

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

def gcn_test(gcn, graph_testloader, args:Args):

    def get_acc(y_hat, y):
        return torch.sum(y_hat == y) / len(y)

    accs = []
    with torch.no_grad():
        for graph_idx, graph in enumerate(graph_testloader):
            x, y = graph.x.to(args.device), graph.y.to(args.device)
            edge_index, edge_weight = graph.edge_index.to(args.device), graph.edge_weight.to(args.device)
            y_hat = gcn(x, edge_index, edge_weight).argmax(axis=-1)
            accs.append(get_acc(y_hat, y).item())
            print(f"在第 {graph_idx+1} 张测试图中, 测试正确率为:{accs[-1]*100:.2f}%")
            
    print(f"在测试集上的平均正确率:{sum(accs) / len(accs)*100:.2f}%")
    fig = plt.figure()
    plt.plot(list(range(1, len(graph_testloader)+1)), accs)
    plt.xlabel('Graph')
    plt.ylabel('Accurancy')
    plt.title('Test Accurancy For Graphs')
    res_folder = f"./Results/{args.N}-way-{args.K}-shots"
    if not os.path.exists(res_folder):
        os.makedirs(res_folder)
    plt.savefig(os.path.join(res_folder, f"{args.SNR}SNR-gcn-test-res-{args.N}-way-{args.K}-shots-"+args.transform))

def overall_process(mimii_dataset_SNR, N, K, query_size, train_test_ratio,
                    cnn_num_hidden_channels, cnn_lr, cnn_num_epochs,
                    gcn_embed_size, gcn_lr, gcn_num_epochs, dataset_transform=None):
    """
    整个图卷积的全过程\n
    Dataset -> DataLoader --Feed to CNN -> feature_vectors -> GraphDataLoader
    --Feed to GCN -> pred_classe

    cnn_type : 可选 默认'1D', 可选['1D', '2D']
    """
    args = Args(seed=42, N=N, K=K, query_size=query_size, SNR=mimii_dataset_SNR, transform=dataset_transform)

    if args.device != "cpu":
        num_workers = 1
        pin_memory = True
        torch.cuda.manual_seed(args.seed)
    else:
        num_workers = 0
        pin_memory = False
        torch.manual_seed(args.seed)

    # 加载数据集
    mimii_dataset = datasets.MIMII(root_dir=f'./data/mimii/{mimii_dataset_SNR}'+'dB_SNR',
                                   N=N, K=K, query_size=query_size, transform=dataset_transform)
    print(f"Length of Dataset: {len(mimii_dataset)}")
    train_size = int(train_test_ratio * len(mimii_dataset))
    test_size = len(mimii_dataset) - train_size
    train_set, test_set = random_split(mimii_dataset, [train_size, test_size])
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    print(f"Train Size:{train_size}  |  Test Size:{test_size}")

    # 可视化一个样本
    fig = plt.figure()
    waveform_shape = 0
    for support_set, _ in train_loader:
        for waveform, label in support_set:
            waveform = waveform.squeeze(0)
            print("波形形状: {}".format(waveform.size()))
            print("波形采样率: {}".format(mimii_dataset.sample_rate))
            if len(waveform.shape) == 3: # 转换过的2D波形
                plt.imshow(waveform[0], origin='lower')
                plt.colorbar(label="Decibels (dB)")
            else: # 1D 纯波形
                plt.plot(waveform[0].T.numpy())
            waveform_shape = waveform.shape
            break
        break

    # CNN训练
    if dataset_transform is None:
        cnn = nets.AudioCNN(num_channels_input=waveform_shape[0], 
                            num_channels_hidden=cnn_num_hidden_channels, 
                            num_classes_output=args.N)
    else:
        cnn = nets.AudioCNN2D(num_channels_input=waveform_shape[0], 
                            num_channels_hidden=cnn_num_hidden_channels, 
                            num_classes_output=args.N)
    cnn_train(cnn, train_loader, num_epochs=cnn_num_epochs, 
              lr=cnn_lr, args=args)
    
    # 将特征向量转为图结构
    graphs = to_graph_dataset(cnn, train_loader, device=args.device)
    graph_dataset = datasets.GraphDataset(graphs=graphs)
    query_size = mimii_dataset.query_size
    graph_trainloader = GeometricDataLoader(graph_dataset, batch_size=1, shuffle=True)
    print(f"图训练数据集中共有{len(graph_trainloader)}张图")

    # GCN训练
    gcn = nets.GCN(num_features_inputs=cnn_num_hidden_channels*2, 
                   embed_size=gcn_embed_size, num_classes_output=args.N)
    gcn_train(gcn, graph_trainloader, query_size,
              num_epochs=gcn_num_epochs, lr=gcn_lr, args=args)
    
    # GCN测试
    test_graphs = to_graph_dataset(cnn, test_loader, device=args.device)
    graph_testset = datasets.GraphDataset(graphs=test_graphs)
    graph_testloader = GeometricDataLoader(graph_testset, batch_size=1, shuffle=False)
    print(f"图测试数据集中共有{len(graph_testloader)}张图")
    gcn_test(gcn, graph_testloader, args)
    
