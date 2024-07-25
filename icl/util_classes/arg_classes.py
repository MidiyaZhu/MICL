import os
import pickle
import warnings
from dataclasses import field, dataclass
from typing import List, Optional
from ..project_constants import FOLDER_ROOT


def set_default_to_empty_string(v, default_v, activate_flag):
    if ((default_v is not None and v == default_v) or (
            default_v is None and v is None)) and (activate_flag):
        return ""
    else:
        return f'_{v}'


@dataclass
class DeepArgs:
    task_name: str = "sst2"
    model_name: str = "gpt2-xl"
    seeds: List[int] = field(default_factory=lambda: [42])#42,43,44,45,46
    sample_size: int = 1000
    demonstration_shot: int = 1
    split: bool = False
    demonstration_from: str = 'train'
    demonstration_total_shot: int = None
    sample_from: str = 'test'
    device: str = 'cuda:0'
    batch_size: int = 1
    save_folder: str = os.path.join(FOLDER_ROOT, 'results', 'deep')
    using_old: bool = False

    @property
    def save_file_name(self):
        file_name = (
            f"{self.task_name}_{self.model_name}_{self.demonstration_shot}_{self.demonstration_from}"
            f"_{self.sample_from}_{self.sample_size}_{'_'.join([str(seed) for seed in self.seeds])}")
        file_name += set_default_to_empty_string(self.demonstration_total_shot, None,
                                                 self.using_old)
        file_name = os.path.join(self.save_folder, file_name)
        return file_name

    def __post_init__(self):
        assert self.demonstration_from in ['train']
        assert self.sample_from in ['test','mydataset']
        # assert self.label_idx_num in [1,2]
        assert self.task_name in ['sst2', 'agnews', 'trec', 'emo','isear','aman','cr','glue-sst2100','ag_news100','imdb','ag_news']
        assert self.model_name in ['gpt2-xl', 'gpt-j-6b','llama-2-7b-chat-hf', "meta-llama/Llama-2-7b-chat-hf","meta-llama/Meta-Llama-3-8B"]
        assert 'cuda:' in self.device
        self.gpu = int(self.device.split(':')[-1])
        self.actual_sample_size = self.sample_size

        if self.task_name == 'sst2':
            label_dict = {0: ' Negative', 1: ' Positive'}
        elif self.task_name == 'cr':
            label_dict = {0: ' Negative', 1: ' Positive'}
        elif self.task_name == 'imdb':
            label_dict = {0: ' Negative', 1: ' Positive'}
        elif self.task_name == 'glue-sst2100':
            label_dict = {0: ' negative', 1: ' positive'}
        elif self.task_name == 'agnews' or self.task_name == 'ag_news100' or self.task_name == 'ag_news':
            label_dict = {0: ' Worlds', 1: ' Sports', 2: ' Business', 3: ' Technology'}
        elif self.task_name == 'trec':
            label_dict = {0: ' Abbreviation', 1: ' Entity', 2: ' Description', 3: ' Person',
                          4: ' Location',
                          5: ' Number'}
        elif self.task_name == 'emo':
            label_dict = {0: ' Others', 1: ' Happy', 2: ' Sad', 3: ' Angry'}
        elif self.task_name == 'isear':
            label_dict = {0: ' fear', 1: ' sadness', 2: ' disgust', 3: ' anger',4: 'joy', 5:'guilt',6:'shame'}
        elif self.task_name == 'aman':
            label_dict = {0: ' fear', 1: ' sadness', 2: ' disgust', 3: ' anger', 4: 'joy', 5: 'surprise', 6: 'others'}
        else:
            raise NotImplementedError(f"task_name: {self.task_name}")
        self.label_dict = label_dict


    def load_result(self):
        with open(self.save_file_name, 'rb') as f:
            return pickle.load(f)


@dataclass
class ReweightingArgs(DeepArgs):
    save_folder: str = os.path.join(FOLDER_ROOT, 'results', 'reweighting')
    lr: float = 0.1
    train_num_per_class: int = 4
    epoch_num: int = 10
    n_head: int = 32  # 25
    # label_idx_num: int = 2
    # randint: str = '2'
    # def __post_init__(self):
    #     super(ReweightingArgs, self).__post_init__()
    #     save_folder = os.path.join(self.save_folder,
    #                                f"lr_{self.lr}_train_num_{self.train_num_per_class}_epoch_{self.epoch_num}"
    #                                f"_nhead_{self.n_head}_{self.label_idx_num}_{self.randint}")
    #
    #     self.save_folder = save_folder

    def __post_init__(self):
        super(ReweightingArgs, self).__post_init__()
        self.update_save_folder()

    def update_save_folder(self):
        self.save_folder = os.path.join(self.save_folder,
                                        f"lr_{self.lr}_train_num_{self.train_num_per_class}_epoch_{self.epoch_num}"
                                        f"_nhead_{self.n_head}")


@dataclass
class CompressArgs(DeepArgs):
    save_folder: str = os.path.join(FOLDER_ROOT, 'results', 'compress')

@dataclass
class CompressTopArgs(DeepArgs):
    ks_num: int = 20
    save_folder: str = os.path.join(FOLDER_ROOT, 'results', 'compress_top')


@dataclass
class CompressTimeArgs(DeepArgs):
    save_folder: str = os.path.join(FOLDER_ROOT, 'results', 'compress_time')


@dataclass
class AttrArgs(DeepArgs):
    save_folder: str = os.path.join(FOLDER_ROOT, 'results', 'attr')


@dataclass
class ShallowArgs(DeepArgs):
    mask_layer_num: int = 5
    mask_layer_pos: str = 'first'  # first, last
    save_folder: str = os.path.join(FOLDER_ROOT, 'results', 'shallow')

    @property
    def save_file_name(self):
        file_name = (
            f"{self.task_name}_{self.model_name}_{self.demonstration_shot}_{self.demonstration_from}"
            f"_{self.sample_from}_{self.sample_size}_{'_'.join([str(seed) for seed in self.seeds])}"
            f'_{self.mask_layer_num}_{self.mask_layer_pos}')
        file_name += set_default_to_empty_string(self.demonstration_total_shot, None,
                                                 self.using_old)

        file_name = os.path.join(self.save_folder, file_name)
        return file_name

    def __post_init__(self):
        super().__post_init__()
        assert self.mask_layer_pos in ['first', 'last']
        if self.mask_layer_num < 0:
            warnings.warn(f"mask_layer_num: {self.mask_layer_num} < 0!")


@dataclass
class NClassificationArgs(DeepArgs):
    save_folder: str = os.path.join(FOLDER_ROOT, 'results', 'nclassfication')

@dataclass
class ShallowNonLabelArgs(ShallowArgs):
    save_folder: str = os.path.join(FOLDER_ROOT, 'results', 'shallow_non_label')