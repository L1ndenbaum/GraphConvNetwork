import torch, random, torchaudio, os
import pyreadr
import pandas as pd
from torch.utils.data import Dataset
from torchaudio import transforms as transforms

def random_overload(class_samples:dict, classes:list):
    # 从样本里随机重新采样以平衡数据集
    class_maxlen = max([len(samples) for samples in class_samples.values()])
    for class_id in classes:
        num_oversampled = class_maxlen - len(class_samples[class_id])
        if num_oversampled > 0:
            class_samples[class_id].extend(random.choices(class_samples[class_id], 
                                                          k=num_oversampled))

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

    def __init__(self, root_dir, N, K, query_size, is_train, train_test_ratio:float=None, num_trainsets:int=None,
                 machine_classes=['fan', 'pump', 'slider', 'valve'], 
                 model_ids=['id_00', 'id_02', 'id_04', 'id_06'], categories=['normal', 'abnormal'],
                 resample=False, resample_rate=None, seed=42):
        super().__init__()
        assert any([num_trainsets is not  None, train_test_ratio is not None]), "必须只使用两个参数中的一个"
        assert not all([num_trainsets is not  None, train_test_ratio is not None]), "必须只使用两个参数中的一个"
        random.seed(seed)
        self.train_test_ratio = train_test_ratio
        self.num_trainsets = num_trainsets
        self.name = 'mimii'
        self.SNR = root_dir.split('/')[-1][0]
        self.sample_rate = 16000
        self.num_channels = 8
        self.root_dir = root_dir
        self.is_train = is_train
        self.machine_classes = machine_classes  # ['fan', 'pump', 'slider', 'valve']
        self.model_ids = model_ids  # ['id_00', 'id_02', 'id_04', 'id_06']
        self.categories = categories  # ['normal', 'abnormal']
        self.resample = resample
        self.resample_rate = resample_rate
        self.N, self.K, self.query_size = N, K, query_size
        self.class_samples = self._load_class_samples() # 结构: {机械种类_机械型号_是否故障: [音频路径]}
        self.classes = list(self.class_samples.keys()) # 共32种 如: 'fan_id_00_normal'
        self.labels_to_indexes = {class_str:i for i, class_str in enumerate(self.classes)}
        self.index_to_labels = {i:class_str for i, class_str in enumerate(self.classes)}
        self.in_data_type = 'audio'
        self.transform = transforms.Spectrogram(n_fft=1024, 
                                                win_length=64, 
                                                hop_length=256,
                                                power=2)

    def _load_class_samples(self):
        class_samples = {}
        training_class_samples = {}
        testing_class_samples = {}

        for machine_class in self.machine_classes:
            for model in self.model_ids:
                for category in self.categories:
                    class_id = f'{machine_class}_{model}_{category}'
                    category_dir = os.path.join(self.root_dir, machine_class, machine_class, model, category)
                    class_samples[class_id] = [os.path.join(category_dir, f) for f in os.listdir(category_dir) if f.endswith('.wav')]
                    if self.is_train:
                        if self.train_test_ratio is not None:
                            training_class_samples[class_id] = class_samples[class_id][:int(len(class_samples[class_id])*self.train_test_ratio)]
                        else:
                            training_class_samples[class_id] = class_samples[class_id][:self.num_trainsets*(self.K+self.query_size)]
                    else:
                        if self.train_test_ratio is not None:
                            testing_class_samples[class_id] = class_samples[class_id][int(len(class_samples[class_id])*self.train_test_ratio):]
                        else:
                            testing_class_samples[class_id] = class_samples[class_id][self.num_trainsets*(self.K+self.query_size):]

        # 从样本里随机重新采样以平衡数据集
        if self.is_train:
            random_overload(training_class_samples, training_class_samples.keys())
            return training_class_samples
        else:
            random_overload(testing_class_samples, testing_class_samples.keys())
            return testing_class_samples

    def __len__(self):
        return int(max([len(self.class_samples[class_id]) for class_id in self.classes]) / (self.K+self.query_size))

    def __getitem__(self, idx):
        support_waveforms, support_labels = [], []

        for class_id in self.classes:
            for k in range(self.K):
                support_file_path = self.class_samples[class_id][self.K*idx + k]
                waveform, sample_rate = torchaudio.load(support_file_path)
                if self.resample:
                    waveform = torchaudio.transforms.Resample(sample_rate, self.resample_rate)(waveform)
                support_waveforms.append(waveform)
                support_labels.append(torch.tensor(self.labels_to_indexes[class_id], dtype=torch.long))

        # 打乱 SupportSet
        paired = list(zip(support_waveforms, support_labels))
        random.shuffle(paired)
        support_waveforms, support_labels = zip(*paired)
        support_waveforms = list(support_waveforms)
        support_labels = list(support_labels)

        query_waveforms, query_labels = [], []
        for class_id in self.classes:
            for query_idx in range(self.query_size):
                query_file_path = self.class_samples[class_id][self.K*idx + self.K + query_idx]
                waveform, sample_rate = torchaudio.load(query_file_path)
                if self.resample:
                    waveform = torchaudio.transforms.Resample(sample_rate, self.resample_rate)(waveform)
                query_waveforms.append(waveform)
                query_labels.append(torch.tensor(self.labels_to_indexes[class_id], dtype=torch.long))

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
        return f"Label-to-index --> {self.labels_to_indexes}"
    
class TennesseeEastman(Dataset):
    """TE化工数据集"""
    def __init__(self, root_dir, N, K, query_size, is_training=False, selected_faults=[i+1 for i in range(8)], seed=42):
        super().__init__()
        random.seed(seed)
        self.name = 'te'
        self.in_data_type = 'seq'
        self.N, self.K = N, K
        self.query_size = query_size
        self.is_training = is_training
        self.selected_faults = selected_faults
        self.class_samples = self.read_data(root_dir=root_dir)
        self.labels_to_indexes = {class_str:i for i, class_str in enumerate(selected_faults)}
        self.index_to_labels = {i:class_str for i, class_str in enumerate(selected_faults)}

    def __len__(self):
        return int(max([len(class_samples) for class_samples in self.class_samples.values()]) / (self.K + self.query_size))
    
    def __getitem__(self, idx):
        support_waveforms, support_labels = [], []

        for fault_number in self.selected_faults:
            for k in range(self.K):
                waveform = self.class_samples[fault_number][self.K*idx + k]
                support_waveforms.append(waveform)
                support_labels.append(torch.tensor(self.labels_to_indexes[fault_number], dtype=torch.long))

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
                query_labels.append(torch.tensor(self.labels_to_indexes[fault_number], dtype=torch.long))

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
    
class XJTUBearing(Dataset):
    def __init__(self, root_dir, condition:int, selected_faults:list, K:int, query_size:int, 
                 is_train:bool, train_test_ratio:float=None, num_trainsets:int=None):
        super().__init__()
        assert condition in {1, 2, 3}, "参数condition必须是1或2或3"
        assert all([0<=fault<=5 for fault in selected_faults]), "选择的fault数字代号必须在1到5之内"
        assert any([num_trainsets is not None, train_test_ratio is not None]), "必须只使用两个参数中的一个"
        assert not all([num_trainsets is not None, train_test_ratio is not None]), "必须只使用两个参数中的一个"
        self.name = 'xjtu_bearing'
        self.in_data_type = 'audio'
        self.condition_str_dict = {1 : '35Hz12kN',
                                   2 : '37.5Hz11kN',
                                   3 : '40Hz10kN'}
        self.root_dir = root_dir
        self.is_train = is_train
        self.condition = condition
        self.train_test_ratio = train_test_ratio
        self.num_trainsets = num_trainsets
        self.N = len(selected_faults)
        self.K = K
        self.query_size = query_size
        self.selected_faults = selected_faults
        self.class_samples = self._load_class_samples()
        self.label_name_dict = self.get_label_name_dict(condition)
        self.labels_to_indexes = {class_str:i for i, class_str in enumerate(selected_faults)}
        self.index_to_labels = {i:class_str for i, class_str in enumerate(selected_faults)}
        self.transform = transforms.Spectrogram(n_fft=4096, 
                                                win_length=64, 
                                                hop_length=256,
                                                power=2)

    def get_label_name_dict(self, condition:int):
        if condition == 1:
            return {1 : 'Outer Race',
                    2 : 'Outer Race',
                    3 : 'Outer Race',
                    4 : 'Cage',
                    5 : 'Inner Race, Outer Race'}
        elif condition == 2:
            return {1 : 'Inner Race',
                    2 : 'Outer Race',
                    3 : 'Cage',
                    4 : 'Outer Race',
                    5 : 'Outer Race'}
        elif condition == 3:
            return {1 : 'Outer Race',
                    2 : 'Inner Race, Ball, Cage, Outer Race',
                    3 : 'Inner Race',
                    4 : 'Inner Race',
                    5 : 'Outer Race'}
        
    def _load_class_samples(self):
        class_samples = {_class : []  for _class in self.selected_faults}
        training_class_samples = {_class : []  for _class in self.selected_faults}
        testing_class_samples = {_class : []  for _class in self.selected_faults}

        for fault_id in self.selected_faults:
            fault_files_path = os.path.join(self.root_dir, self.condition_str_dict[self.condition],
                                            f'Bearing{self.condition}_{fault_id}')
            class_fault_csv_list = [pd.read_csv(os.path.join(fault_files_path, f)).to_numpy().T for f in os.listdir(fault_files_path) 
                                    if f.endswith('.csv')]
            class_samples[fault_id] = [torch.tensor(csv, dtype=torch.float32) for csv in class_fault_csv_list]
            if self.is_train:
                if self.train_test_ratio is not None:
                    training_class_samples[fault_id] = class_samples[fault_id][:int(len(class_samples[fault_id])*self.train_test_ratio)]
                else:
                    training_class_samples[fault_id] = class_samples[fault_id][:self.num_trainsets*(self.K+self.query_size)]
            else:
                if self.train_test_ratio is not None:
                    testing_class_samples[fault_id] = class_samples[fault_id][int(len(class_samples[fault_id])*self.train_test_ratio):]
                else:
                    testing_class_samples[fault_id] = class_samples[fault_id][self.num_trainsets*(self.K+self.query_size):]

        # 从样本里随机重新采样以平衡数据集
        if self.is_train:
            random_overload(training_class_samples, training_class_samples.keys())
            return training_class_samples
        else:
            random_overload(testing_class_samples, testing_class_samples.keys())
            return testing_class_samples

    def __len__(self):
        return int(max([len(class_samples) for class_samples in self.class_samples.values()]) / (self.K + self.query_size))
    
    def __getitem__(self, idx):
        support_waveforms, support_labels = [], []
        for fault_number in self.selected_faults:
            for k in range(self.K):
                waveform = self.class_samples[fault_number][self.K*idx + k]
                support_waveforms.append(waveform)
                support_labels.append(torch.tensor(self.labels_to_indexes[fault_number], dtype=torch.long))

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
                query_labels.append(torch.tensor(self.labels_to_indexes[fault_number], dtype=torch.long))

        # 打乱 QuerySet
        paired = list(zip(query_waveforms, query_labels))
        random.shuffle(paired)
        query_waveforms, query_labels = zip(*paired)
        query_waveforms = list(query_waveforms)
        query_labels = list(query_labels)

        waveforms = torch.stack(support_waveforms+query_waveforms)
        spectrograms = self.transform(waveforms)
        labels = torch.stack(support_labels+query_labels)

        return waveforms, spectrograms, labels

