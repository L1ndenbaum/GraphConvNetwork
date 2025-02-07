import torch, os, time, random, numpy
import matplotlib.pyplot as plt
from torch_geometric.utils import dense_to_sparse

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
    def __init__(self, seed, N, K, query_size, dataset_name, in_data_type, SNR=None):
        self.device = try_gpu()
        self.seed = seed
        self.SNR = SNR
        self.N = N
        self.K = K
        self.query_size = query_size
        self.dataset_name = dataset_name
        self.in_data_type = in_data_type

class TestInfo:
    def __init__(self):
        raise NotImplementedError
        self.dataset_name = dataset_name
        self.dataset_type = 'seq'

def try_gpu(i=0):
    """如果存在,返回gpu(i), 否则返回cpu"""
    if torch.cuda.device_count() >= i+1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def global_manual_seed(seed, device):
    random.seed(seed)          # Python's random module
    numpy.random.seed(seed)       # NumPy
    torch.manual_seed(seed)    # PyTorch CPU
    if device != "cpu":
        torch.cuda.manual_seed(seed)
    else:
        torch.manual_seed(seed)

def save_results(args:Args, avg_test_acc, results_dir='./Results'):
    """
    记录在测试集上实验结果的函数
    记录一组实验结果格式为:
    N K 平均test_acc
    """
    results_dir = os.path.join(results_dir, args.dataset)
    if args.dataset == 'mimii':
        results_dir = os.path.join(results_dir, f'{args.SNR}SNR')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'accuracy-records.txt'), 'a') as rec_file:
        rec_file.write(f"{args.N}\t{args.K}\t{avg_test_acc}\n")

def read_results(datasets, results_dir='./Results', mimii_snr=None):
    raise NotImplementedError
    results_dir = os.path.join(results_dir, dataset)
    if not os.path.exists(results_dir):
        raise ValueError(f"目前{dataset}数据集还没有实验结果")
    if dataset == 'mimii':
        results_dir = os.path.join(results_dir, f'{SNR}SNR')
    with open(os.path.join(results_dir, 'accuracy-records.txt'), 'a') as rec_file:
        for line in rec_file:
            N, K, avg_test_acc = line.strip().split('\t')
            N, K, avg_test_acc = int(N), int(K), float(avg_test_acc)

def in_class_adj_matrix(labels:torch.Tensor):
    """
    Args:\n
        labels : 样本节点对应的标签\n
    Returns:\n
        edge_index : 一个torch_geometric下的edge_index
    返回的edge_index让所有属于同一类的节点彼此连接(带自旋), 不属于同类的节点无连接
    """
    num_nodes = labels.shape[0]
    adj = torch.zeros((num_nodes, num_nodes))
    for node_idx in range(num_nodes):
        indices = (labels == labels[node_idx]).nonzero()
        adj[node_idx, indices] = 1
    edge_index, _ = dense_to_sparse(adj)
    edge_index = edge_index.to(labels.device)

    return edge_index

def cross_class_adj_matrix(labels:torch.Tensor):
    """
    Args:\n
        labels : 样本节点对应的标签\n
    Returns:\n
        edge_index : 一个torch_geometric下的edge_index
    返回的edge_index让所有属于不同类的节点彼此连接(无自旋), 属于同类的节点无连接
    """
    num_nodes = labels.shape[0]
    adj = torch.zeros((num_nodes, num_nodes))
    for node_idx in range(num_nodes):
        indices = (labels != labels[node_idx]).nonzero()
        adj[node_idx, indices] = 1
    edge_index, _ = dense_to_sparse(adj)
    edge_index = edge_index.to(labels.device)

    return edge_index

def final_pooling_adj_matrix(labels, num_queries):
    """
    Args:\n
        lables : 样本节点对应的标签\n
        num_queries : QuerySet中的节点总数\n
    Returns:\n
        edge_index : 一个torch_geometric下的edge_index
    返回的edge_index让 SupportSet 中所有属于同类的节点彼此连接(带自旋), 属于同类的节点无连接
    对于 QuerySet 中的节点, 每一个Query节点都与其他所有节点连接(不带自旋)
    """
    num_nodes = labels.shape[0]
    adj = torch.zeros((num_nodes, num_nodes))
    for node_idx in range(num_nodes-num_queries):
        indices = (labels == labels[node_idx]).nonzero()
        adj[node_idx, indices] = 1
    for node_idx in range(num_nodes-num_queries, num_nodes):
        adj[node_idx, :] = 1
    adj[num_nodes-num_queries:, :].fill_diagonal_(0)
    edge_index, _ = dense_to_sparse(adj)
    edge_index = edge_index.to(labels.device)

    return edge_index