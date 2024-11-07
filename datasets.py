import torch.nn as nn
import torch.optim as optim
import torch, random, torchaudio, os, torch_geometric
from torch.utils.data import Dataset
from torchvision import transforms as transforms
from torch_geometric.data import Dataset as GeometricDataset

class MIMII(Dataset):
    """
    用于小样本学习的MIMII数据集
    """
    def __init__(self, root_dir, N, K, query_size, machine_classes=['fan', 'pump', 'slider', 'valve'], 
                 model_ids=['id_00', 'id_02', 'id_04', 'id_06'], categories=['normal', 'abnormal'],
                 resample=False, resample_rate=None):
        super().__init__()
        self.root_dir = root_dir
        self.machine_classes = machine_classes  # ['fan', 'pump', 'slider', 'valve']
        self.model_ids = model_ids  # ['id_00', 'id_02', 'id_04', 'id_06']
        self.categories = categories  # ['normal', 'abnormal']
        self.resample = resample
        self.resample_rate = resample_rate
        self.data = self._load_class_list() # 结构: {机械种类_机械型号_是否故障: [音频路径]}
        self.classes = list(self.data.keys()) # 共32种 如: 'fan_id_00_normal'
        self.labels_to_indexs = {class_str:i for i, class_str in enumerate(self.classes)}
        self.index_to_labels = {i:class_str for i, class_str in enumerate(self.classes)}
        self.N, self.K, self.query_size = N, K, query_size
        self.N_classes = random.sample(self.classes, k=N)

    def _load_class_list(self):
        class_list = {}
        for machine_class in self.machine_classes:
            for model in self.model_ids:
                for category in self.categories:
                    class_id = f'{machine_class}_{model}_{category}'
                    class_list[class_id] = []
                    category_dir = os.path.join(self.root_dir, machine_class, machine_class, model, category)
                    class_list[class_id] = [os.path.join(category_dir, f) for f in os.listdir(category_dir) if f.endswith('.wav')]

        return class_list

    def __len__(self):
        return int(min([len(self.data[class_id]) for class_id in self.N_classes]) / self.K)

    def __getitem__(self, idx):
        support_set, query_set = [], []

        for class_id in self.N_classes:
            for k in range(self.K):
                if self.K*idx+k < len(self.data[class_id]):
                    file_path = self.data[class_id][self.K*idx+k]
                    waveform, sample_rate = torchaudio.load(file_path)
                    if self.resample:
                        waveform = torchaudio.transforms.Resample(sample_rate, self.resample_rate)(waveform)
                    support_set.append((waveform, self.labels_to_indexs[class_id]))

        for _ in range(self.query_size):
            query_class_id = random.choice(self.N_classes)
            query_file_path = random.choice(self.data[query_class_id])
            query_waveform, sample_rate = torchaudio.load(query_file_path)
            if self.resample:
                    query_waveform = torchaudio.transforms.Resample(sample_rate, self.resample_rate)(query_waveform)

            query_set.append((query_waveform, self.labels_to_indexs[query_class_id]))
            
        random.shuffle(support_set)
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
    