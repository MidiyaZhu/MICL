from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd
from src.utils.app import App
from src.dataset_readers.dataset_wrappers.base import *
from src.datasets.labels import get_mapping_token

field_getter = App()

# e.g. label2text = {0: "negative", 1: "positive"}
label2text = get_mapping_token("aman")


@field_getter.add("X")
def get_X(entry):
    return entry['text'] if 'X' not in entry.keys() else entry['X']


@field_getter.add("Y_TEXT")  # 获得标签对应的文本
def get_Y_TEXT(entry):
    return label2text[entry['label']] if 'Y_TEXT' not in entry.keys() else entry['Y_TEXT']


@field_getter.add("C")  # 数据集有sentence1,sentence2,label 时，用C表示sentence1
def get_C(entry):
    return "" if 'C' not in entry.keys() else entry['C']


@field_getter.add("Y")  # 获得原始标签 int
def get_Y(entry):
    return entry['label'] if 'Y' not in entry.keys() else entry['Y']


@field_getter.add("ALL")
def get_ALL(entry):
    return f"{entry['text']}\tIt is {label2text[entry['label']]}" if 'ALL' not in entry.keys() else entry['ALL']


class AMANDatasetWrapper(DatasetWrapper):
    name = "aman"
    sentence_field = "text"
    label_field = "label"

    def __init__(self, dataset_path=None, dataset_split=None, ds_size=None):
        def _abs_label(ex):
            ex['label'] = abs(ex['label'])
            return ex

        super().__init__()
        self.task_name = "aman"
        self.field_getter = field_getter
        self.postfix = "It is"  # for inference
        # if dataset_path is None:
        #     self.dataset = load_dataset("gpt3mix/sst2", split=dataset_split)
        #     self.dataset = self.dataset.map(_abs_label, batched=False, load_from_cache_file=False)
        # else:

        # dftrain = pd.read_csv('/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/self-adaptive-ICL/self-adaptive-ICL-main/data/train_isear_joy.csv')  # replace with your actual path
        # dftest = pd.read_csv('/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/self-adaptive-ICL/self-adaptive-ICL-main/data/test_isear.csv')  # replace with your actual path
        # dataset_trian = Dataset.from_pandas(dftrain)
        # dataset_test = Dataset.from_pandas(dftest)
        if dataset_split == 'test':
            # dftest = pd.read_csv('/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/self-adaptive-ICL/self-adaptive-ICL-main/data/test_aman.csv')  # replace with your actual path
            # dftest = pd.read_csv('/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/self-adaptive-ICL/self-adaptive-ICL-main/data/valid/valid_aman.csv')  # replace with your actual path
            dftest = pd.read_csv(dataset_path)
            dataset_test = Dataset.from_pandas(dftest)
            self.dataset = dataset_test
        elif dataset_split == 'train':
            dftrain = pd.read_csv(dataset_path)  # replace with your actual path
            dataset_trian = Dataset.from_pandas(dftrain)
            self.dataset = dataset_trian
        # self.dataset = Dataset.from_pandas(pd.DataFrame(data=pd.read_csv(dataset_path)))
        elif dataset_split == 'retrieve':
            self.dataset = Dataset.from_pandas(pd.DataFrame(data=pd.read_json(dataset_path)))

        if ds_size is not None:
            self.dataset = load_partial_dataset(self.dataset, size=ds_size)
