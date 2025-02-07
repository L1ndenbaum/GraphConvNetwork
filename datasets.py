import torch, random, torchaudio, os
from torch.utils.data import Dataset
from torchaudio import transforms as transforms
import pyreadr

class MIMII(Dataset):
    """
    用于小样本学习的MIMII数据集\n
    参数:\n
        root_dir : MIMII数据集的根目录\n
        N : N个支持类别\n
        K : 为每个支持类别提供K个支持样本\n
        machine_classes : 机器的种类, 最多四种, 默认全选(['fan', 'pump', 'slider', 'valve'])\n
        model_ids : 每种机器的型号, 最多四种, 默认全选(['id_00', 'id_02', 'id_04', 'id_06'])\n
        categories : 机器是否故障, 默认全选(['normal', 'abnormal'])\n
        indicated_query_classes : 默认无, 可手动提供, 表示希望将这些类别作为询问类别\n
        transform : 将波形转换为2D Image的变换方式, 目前只提供:['spectrogram']\n
        resample : 是否对.wav波形重采样, 默认False, 如果使用True需提供重采样率\n
        resample_rate : 重采样率\n
        seed : 用于初始化random库的种子

    __getitem__()返回值:\n
        每次返回支持集和询问集, 每个集中的每个元素为三元组(Raw波形, STFT_Image, label)
    """

    def __init__(self, root_dir, N, K, query_size, machine_classes=['fan', 'pump', 'slider', 'valve'], 
                 model_ids=['id_00', 'id_02', 'id_04', 'id_06'], categories=['normal', 'abnormal'],
                 indicated_query_classes=None, transform='spectrogram', resample=False, resample_rate=None, seed=42):
        super().__init__()
        random.seed(seed)
        self.name = 'mimii'
        self.SNR = root_dir.split('/')[-1][0]
        self.sample_rate = 16000
        self.num_channels = 8
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
        self.in_data_type = 'audio'
        if transform == 'spectrogram':
            self.transform = transforms.Spectrogram(n_fft=1024, 
                                                    win_length=64, 
                                                    hop_length=256,
                                                    power=2)

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
        support_waveforms, support_labels = [], []
        for class_id in self.N_classes:
            for k in range(self.K):
                support_file_path = self.data[class_id][self.K*idx + k]
                waveform, sample_rate = torchaudio.load(support_file_path)
                if self.resample:
                    waveform = torchaudio.transforms.Resample(sample_rate, self.resample_rate)(waveform)
                support_waveforms.append(waveform)
                support_labels.append(torch.tensor(self.labels_to_indexs[class_id], dtype=torch.long))

        # 打乱 SupportSet
        paired = list(zip(support_waveforms, support_labels))
        random.shuffle(paired)
        support_waveforms, support_labels = zip(*paired)
        support_waveforms = list(support_waveforms)
        support_labels = list(support_labels)

        query_waveforms, query_labels = [], []
        for class_id in self.N_classes:
            for query_idx in range(self.query_size):
                query_file_path = self.data[class_id][self.K*idx + self.K + query_idx]
                waveform, sample_rate = torchaudio.load(query_file_path)
                if self.resample:
                    waveform = torchaudio.transforms.Resample(sample_rate, self.resample_rate)(waveform)
                query_waveforms.append(waveform)
                query_labels.append(torch.tensor(self.labels_to_indexs[class_id], dtype=torch.long))

        # 打乱 QuerySet
        paired = list(zip(query_waveforms, query_labels))
        random.shuffle(paired)
        query_waveforms, query_labels = zip(*paired)
        query_waveforms = list(query_waveforms)
        query_labels = list(query_labels)
        
        waveforms = torch.stack(support_waveforms+query_waveforms)
        labels = torch.stack(support_labels+query_labels)
        spectrograms = self.transform(waveforms)

        return waveforms, spectrograms, labels

    def __repr__(self):
        return f"Label-to-index --> {self.labels_to_indexs}"
    
class TennesseeEastman(Dataset):
    """TE化工数据集"""
    def __init__(self, root_dir, N, K, query_size, is_training=False, selected_faults=None, seed=42):
        super().__init__()
        random.seed(seed)
        self.name = 'te'
        self.N, self.K = N, K
        self.query_size = query_size
        self.is_training = is_training
        self.selected_faults = selected_faults
        self.class_samples = self.read_data(root_dir=root_dir)
        self.labels_to_indexs = {class_str:i for i, class_str in enumerate(selected_faults)}
        self.index_to_labels = {i:class_str for i, class_str in enumerate(selected_faults)}
        self.transform = transforms.Spectrogram(n_fft=16, hop_length=1, power=1)
        self.in_data_type = 'seq'
        self.sample_max_len = 960

    def __len__(self):
        return int(min([len(class_samples) for class_samples in self.class_samples.values()]) / (self.K + self.query_size))
    
    def __getitem__(self, idx):
        support_waveforms, support_labels = [], []

        for fault_number in self.selected_faults:
            for k in range(self.K):
                waveform = self.class_samples[fault_number][self.K*idx + k]
                support_waveforms.append(waveform)
                support_labels.append(torch.tensor(self.labels_to_indexs[fault_number], dtype=torch.long))

        # 打乱 SupportSet
        paired = list(zip(support_waveforms, support_labels))
        random.shuffle(paired)
        support_waveforms, support_labels = zip(*paired)
        support_waveforms = list(support_waveforms)
        support_labels = list(support_labels)

        query_waveforms, query_labels = [], []
        for fault_number in self.selected_faults:
            for query_idx in range(self.query_size):
                waveform = self.class_samples[fault_number][self.K*idx + self.K + query_idx]
                query_waveforms.append(waveform)
                query_labels.append(torch.tensor(self.labels_to_indexs[fault_number], dtype=torch.long))

        # 打乱 QuerySet
        paired = list(zip(query_waveforms, query_labels))
        random.shuffle(paired)
        query_waveforms, query_labels = zip(*paired)
        query_waveforms = list(query_waveforms)
        query_labels = list(query_labels)

        waveforms = torch.stack(support_waveforms+query_waveforms)
        labels = torch.stack(support_labels+query_labels)

        return waveforms, labels

    def read_data(self, root_dir):
        if self.is_training:
            df = pyreadr.read_r(os.path.join(root_dir, 'TEP_Faulty_Training.RData'))
            data = df['faulty_training'].drop(columns=['simulationRun'])
        else:
            df = pyreadr.read_r(os.path.join(root_dir, 'TEP_Faulty_Testing.RData'))
            data = df['faulty_testing'].drop(columns=['simulationRun'])

        if self.selected_faults is None:
            selected_faults = random.sample(list(range(1, 21)), self.N)
        else:
            selected_faults = self.selected_faults

        data = data[data['faultNumber'].isin(selected_faults)]
        class_groups = [group for _, group in data.groupby('faultNumber')]
        class_samples = {_class : []  for _class in selected_faults}

        points_per_sample = 500 if self.is_training else 960

        for class_group in class_groups:
            class_sample_groups = [group for _, group in class_group.groupby(class_group.index // points_per_sample)]
            for class_sample_group in class_sample_groups:
                features = class_sample_group.drop(columns=['faultNumber', 'sample']).values.T
                label = class_sample_group['faultNumber'].iloc[0]
                class_samples[label].append(torch.tensor(features, dtype=torch.float32))

        return class_samples