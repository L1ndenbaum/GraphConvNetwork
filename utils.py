import torch.nn as nn
import torch.optim as optim
import torch_geometric.utils
import torch, os, time, torch_geometric
import matplotlib.pyplot as plt
from torch_geometric.data import Data

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
            adj_matrix[i][j] = 1 / (torch.sum(torch.sqrt(torch.pow(feature_vecs[i] - feature_vecs[j], 2)), 0)+torch.tensor(1e-5))
    edge_index, edge_weight = torch_geometric.utils.dense_to_sparse(adj_matrix)
    return Data(x=feature_vecs, edge_index=edge_index, edge_weight=edge_weight, y=labels)

def to_graph_dataset(time_cnn:nn.Module, freq_cnn:nn.Module, dataloader:torch.utils.data.DataLoader, args):
    """
    将原始数据集转换为图数据集的函数
    转换后的图数据集将被放置于cpu
    返回:\n
        时域图和频域图
    """
    time_graphs, freq_graphs = [], []
    for support_set, query_set in dataloader:
        waveforms, freq_waveforms, labels = [], [], []
        for waveform, freq_waveform, label in support_set:
            waveforms.append(waveform.squeeze(0))
            freq_waveforms.append(freq_waveform.squeeze(0))
            labels.append(label)
        for waveform, freq_waveform, label in query_set:
            waveforms.append(waveform.squeeze(0))
            freq_waveforms.append(freq_waveform.squeeze(0))
            labels.append(label)
        
        waveforms = torch.stack(waveforms).to(args.device)
        labels = torch.tensor(labels, device=args.device)
        time_feature_vecs = feature_extraction(time_cnn, waveforms)
        del waveforms
        torch.cuda.empty_cache()

        freq_waveforms = torch.stack(freq_waveforms).to(args.device)
        freq_feature_vecs = feature_extraction(freq_cnn, freq_waveforms)
        del freq_waveforms
        torch.cuda.empty_cache()
        
        time_graph = graph_construction(time_feature_vecs, labels)
        freq_graph = graph_construction(freq_feature_vecs, labels)
        time_graphs.append(time_graph)
        freq_graphs.append(freq_graph)

    return time_graphs, freq_graphs

def gcn_train(gcn, time_graph_trainloader, freq_graph_trainloader, query_size, num_epochs, lr, args:Args):

    def init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
        elif isinstance(module, torch_geometric.nn.GCNConv):
            nn.init.xavier_uniform_(module.lin.weight)

    def get_num_correct_query_pred(pred_y, y, query_size):
        return torch.sum(pred_y[-query_size:].argmax(dim=1) == y[-query_size:])

    optimizer = optim.Adam(gcn.parameters(), lr=lr, weight_decay=0.0001) # , weight_decay=0.0001
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.85)
    loss_function = nn.CrossEntropyLoss()
    gcn.to(args.device).train()
    gcn.apply(init_weights)
    _, axes = plt.subplots(1, 2)

    losses, accs = [], []
    for epoch in range(1, num_epochs+1):
        metric = Accumulator(4) # 一个epoch经过样本数 gcn一个epoch所有样本损失总量 一个epoch的query总数 一个epoch正确分类的query总数
        for time_graph, freq_graph in zip(time_graph_trainloader, freq_graph_trainloader):
            time_features, y = time_graph.x.to(args.device), time_graph.y.to(args.device)
            freq_features = freq_graph.x.to(args.device)
            edge_index, time_edge_weight = time_graph.edge_index.to(args.device), time_graph.edge_weight.to(args.device)
            freq_edge_weight = freq_graph.edge_weight.to(args.device)
            optimizer.zero_grad()
            y_hat = gcn(time_features, edge_index, time_edge_weight,
                                 freq_features, freq_edge_weight, y, args.N, args.query_size)
            loss = loss_function(y_hat, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                metric.add(time_features.shape[0], loss.item(),
                           query_size, get_num_correct_query_pred(y_hat, y, query_size))
            del time_features, freq_features, y, edge_index, time_edge_weight, freq_edge_weight
            torch.cuda.empty_cache()
                
        losses.append(metric[1] / metric[0])
        accs.append(metric[3] / metric[2])

        if epoch % 50==0:
            print(f"Epoch:{epoch}  |  GCN_Loss :{losses[-1]:.5f}  |  Query_Acc:{metric[3]/metric[2]*100:.2f}%")

    axes[0].plot(list(range(num_epochs)), losses)
    axes[0].set_title('GCN Training Loss')
    axes[1].plot(list(range(num_epochs)), accs)
    axes[1].set_title('GCN Training Query Accuracy')

    plt.tight_layout()
    res_folder = f"./Results/{args.N}-way-{args.K}-shots"
    if not os.path.exists(res_folder):
        os.makedirs(res_folder)
    plt.savefig(os.path.join(res_folder, f"{args.SNR}SNR-gcn-train-res-{args.N}-way-{args.K}-shots"))
    
def cnn_train(time_cnn, freq_cnn, train_loader, num_epochs, time_cnn_lr, freq_cnn_lr, args:Args):
    """训练时域CNN和频域CNN的函数"""

    def init_weights(module):
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(module.weight)

    time_optimizer = optim.Adam(time_cnn.parameters(), lr=time_cnn_lr)
    freq_optimizer = optim.Adam(freq_cnn.parameters(), lr=freq_cnn_lr)
    time_scheduler = optim.lr_scheduler.StepLR(time_optimizer, step_size=12, gamma=0.95)
    freq_scheduler = optim.lr_scheduler.StepLR(freq_optimizer, step_size=12, gamma=0.75)
    loss_function = nn.CrossEntropyLoss()
    freq_cnn.to(args.device).train()
    freq_cnn.apply(init_weights)
    time_cnn.to(args.device).train()
    time_cnn.apply(init_weights)
    time_losses = []
    freq_losses = []

    for epoch in range(1, num_epochs+1):
        metric = Accumulator(3) # 一个epoch经过样本数 time_cnn一个epoch所有样本损失总量
        for support_set, _ in train_loader:
            waveforms, freq_waveforms, labels = [], [], []
            for waveform, freq_waveform, label in support_set:
                waveforms.append(waveform.squeeze(0))
                freq_waveforms.append(freq_waveform.squeeze(0))
                labels.append(label)
            labels = torch.tensor(labels, device=args.device)
            
            # 时域CNN 训练部分
            waveforms = torch.stack(waveforms).to(args.device)
            time_optimizer.zero_grad()
            time_y_hat = time_cnn(waveforms)
            loss = loss_function(time_y_hat, labels)
            loss.backward()
            time_loss = loss.item()
            time_optimizer.step()
            time_scheduler.step()
            del waveforms

            # 频域CNN 训练部分
            freq_waveforms = torch.stack(freq_waveforms).to(args.device)
            freq_optimizer.zero_grad()
            freq_y_hat = freq_cnn(freq_waveforms)
            loss = loss_function(freq_y_hat, labels)
            loss.backward()
            freq_loss = loss.item()
            freq_optimizer.step()
            freq_scheduler.step()
            metric.add(freq_waveforms.shape[0], time_loss, freq_loss)
            del freq_waveform, labels
            torch.cuda.empty_cache()
        
        time_losses.append(metric[1] / metric[0])
        freq_losses.append(metric[2] / metric[0])
        print(f"Epoch:{epoch}  |  Time_CNN_Loss :{time_losses[-1]:.6f}  |  Freq_CNN_Loss :{freq_losses[-1]:.6f}")
        
    fig = plt.figure()
    plt.plot(list(range(num_epochs)), time_losses, label='Time Loss')
    plt.plot(list(range(num_epochs)), freq_losses, label='Freq Loss')
    plt.title('CNN Loss')
    plt.legend()
    res_folder = f"./Results/{args.N}-way-{args.K}-shots"
    if not os.path.exists(res_folder):
        os.makedirs(res_folder)
    plt.savefig(os.path.join(res_folder, f"{args.SNR}SNR-cnn-train-res-{args.N}-way-{args.K}-shots"))

def feature_extraction(cnn, waveforms):
    """
    用时域CNN和频域CNN提取特征的函数
    返回:\n
        提取后的特征向量, 存放在cpu
    """
    cnn.eval()
    cnn.to(waveforms.device)
    with torch.no_grad():
        feature_vecs = cnn(waveforms)
    return feature_vecs.cpu()

def gcn_test(gcn, time_graph_testloader, freq_graph_testloader, args:Args):
    """测试GCN的函数"""

    def get_acc(y_hat, y):
        return torch.sum(y_hat == y) / len(y)

    accs, query_accs = [], []
    with torch.no_grad():
        for graph_idx, (time_graph, freq_graph) in enumerate(zip(time_graph_testloader, freq_graph_testloader)):
            time_features, y = time_graph.x.to(args.device), time_graph.y.to(args.device)
            freq_features = freq_graph.x.to(args.device)
            edge_index, time_edge_weight = time_graph.edge_index.to(args.device), time_graph.edge_weight.to(args.device)
            freq_edge_weight = freq_graph.edge_weight.to(args.device)
            y_hat = gcn(time_features, edge_index, time_edge_weight,
                                 freq_features, freq_edge_weight, y, args.N, args.query_size).argmax(axis=-1)
            accs.append(get_acc(y_hat, y).item()) # 整个测试图上预测正确率
            query_accs.append(get_acc(y_hat[-args.query_size:], y[-args.query_size:]).item()) # query_set中预测正确率
            print(f"在第 {graph_idx+1} 张测试图中, 整张图测试正确率为:{accs[-1]*100:.2f}%, Query集测试正确率为:{query_accs[-1]*100:.2f}%")
            
    print(f"在整体测试集上的平均正确率:{sum(accs) / len(accs)*100:.2f}%\n在Query测试集的平均正确率为:{sum(query_accs)/len(query_accs)*100:.2f}%")
    fig = plt.figure()
    plt.plot(list(range(1, len(time_graph_testloader)+1)), accs, label='Whole Graph Acc')
    plt.plot(list(range(1, len(time_graph_testloader)+1)), query_accs, label='Query Acc')
    plt.xlabel('Graph')
    plt.ylabel('Accurancy')
    plt.title('Test Accurancy For Graphs')
    plt.legend()
    res_folder = f"./Results/{args.N}-way-{args.K}-shots"
    rec_folder = f"./Records"
    if not os.path.exists(res_folder):
        os.makedirs(res_folder)
    if not os.path.exists(rec_folder):
        os.makedirs(rec_folder)

    # 记录测试结果
    with open(os.path.join(rec_folder, f"{args.SNR}SNR-gcn-test-res-{args.N}-way-{args.K}-shots"), 'a') as rec_file:
        rec_file.write(f"{args.SNR}SNR-gcn-test-res-{args.N}-way-{args.K}-shots\n")
        rec_file.write(f"在整体测试集上的平均正确率:{sum(accs) / len(accs)*100:.2f}%\n在Query测试集的平均正确率为:{sum(query_accs)/len(query_accs)*100:.2f}%\n\n")
    plt.savefig(os.path.join(res_folder, f"{args.SNR}SNR-gcn-test-res-{args.N}-way-{args.K}-shots"))