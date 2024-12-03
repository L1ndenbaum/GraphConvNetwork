import torch.nn as nn
import torch.optim as optim
import torch, random, torchaudio, os, torch_geometric
from torch.utils.data import Dataset
from torchaudio import transforms as transforms
from torch_geometric.data import Dataset as GeometricDataset

class MIMII(Dataset):
    """
    用于小样本学习的MIMII数据集\n
    将整个数据集划分成若干小子集, 在每个子集中, 对每个support_set和query_set,
    对每个类别, 加载前K个作为support_set, 加载后query_size个作为
    参数:\n
        transform : 默认None, 不对wav文件变换, 可选['spectrogram', 'mfcc']
    """
    def __init__(self, root_dir, N, K, query_size, machine_classes=['fan', 'pump', 'slider', 'valve'], 
                 model_ids=['id_00', 'id_02', 'id_04', 'id_06'], categories=['normal', 'abnormal'],
                 indicated_support_samples = None, indicated_query_samples = None,
                 transform=None, resample=False, resample_rate=None):
        super().__init__()
        self.SNR = root_dir.split('/')[-1][0]
        self.sample_rate = 16000
        self.root_dir = root_dir
        self.machine_classes = machine_classes  # ['fan', 'pump', 'slider', 'valve']
        self.model_ids = model_ids  # ['id_00', 'id_02', 'id_04', 'id_06']
        self.categories = categories  # ['normal', 'abnormal']
        self.resample = resample
        self.resample_rate = resample_rate
        self.data = self._load_class_list() # 结构: {机械种类_机械型号_是否故障: [音频路径]}
        self.classes = list(self.data.keys()) # 共32种 如: 'fan_id_00_normal'
        self.N, self.K, self.query_size = N, K, query_size
        self.N_classes = random.sample(self.classes, k=N)
        self.labels_to_indexs = {class_str:i for i, class_str in enumerate(self.N_classes)}
        self.index_to_labels = {i:class_str for i, class_str in enumerate(self.N_classes)}
        self.transform = transform
        if transform == 'spectrogram':
            self.transform = transforms.Spectrogram(
                                            n_fft=4096, win_length=32, hop_length=256, power=1.3
                                            )
        if transform == 'mfcc':
            self.transform = transforms.MFCC(
                                sample_rate=self.sample_rate,
                                n_mfcc=13,                
                                melkwargs={"n_fft": 512,
                                           "hop_length": 256,
                                           "n_mels": 23,
                                           "center": False}
                                )

    def _load_class_list(self):
        class_list = {}
        # 如果指定了
        for machine_class in self.machine_classes:
            for model in self.model_ids:
                for category in self.categories:
                    class_id = f'{machine_class}_{model}_{category}'
                    class_list[class_id] = []
                    category_dir = os.path.join(self.root_dir, machine_class, machine_class, model, category)
                    class_list[class_id] = [os.path.join(category_dir, f) for f in os.listdir(category_dir) if f.endswith('.wav')]

        for machine_class in self.machine_classes:
            for model in self.model_ids:
                for category in self.categories:
                    class_id = f'{machine_class}_{model}_{category}'
                    class_list[class_id] = []
                    category_dir = os.path.join(self.root_dir, machine_class, machine_class, model, category)
                    class_list[class_id] = [os.path.join(category_dir, f) for f in os.listdir(category_dir) if f.endswith('.wav')]

        return class_list

    def __len__(self):
        return int(min([len(self.data[class_id]) for class_id in self.N_classes]) / (self.K+self.query_size))

    def __getitem__(self, idx):
        support_set, query_set = [], []

        for class_id in self.N_classes:
            for k in range(self.K):
                # if self.K*idx+k+self.query_size < len(self.data[class_id]):
                if self.K*idx+k < len(self.data[class_id]):
                    support_file_path = self.data[class_id][self.K*idx+k]
                    support_waveform, sample_rate = torchaudio.load(support_file_path)
                    if self.resample:
                        support_waveform = torchaudio.transforms.Resample(sample_rate, self.resample_rate)(support_waveform)
                    freq_support_waveform = self.transform(support_waveform)
                    support_set.append((support_waveform, freq_support_waveform, self.labels_to_indexs[class_id]))

        query_class_ids = random.sample(self.N_classes, k=self.query_size)
        for query_class_id in query_class_ids:
            for query_idx in range(self.query_size):
                query_file_path = random.choice(self.data[query_class_id])
                #query_file_path = self.data[class_id][self.K*idx+self.K+query_idx]
                query_waveform, sample_rate = torchaudio.load(query_file_path)
                if self.resample:
                    query_waveform = torchaudio.transforms.Resample(sample_rate, self.resample_rate)(query_waveform)
                freq_query_waveform = self.transform(query_waveform)
                query_set.append((query_waveform, freq_query_waveform, self.labels_to_indexs[query_class_id]))

        random.shuffle(support_set)
        random.shuffle(query_set)
        return (support_set, query_set)

    def __repr__(self):
        return f"Label-to-index --> {self.labels_to_indexs}"

class GraphDataset(GeometricDataset):
    def __init__(self, graphs:list[torch_geometric.data.Data]):
        super().__init__()
        self.graphs = graphs

    def len(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return self.graphs[idx]
    