from .movielens_processor import MovielensProcessor
import os
import torch.utils.data as data
import pandas as pd


class Movielens(data.Dataset):
    """`MovieLens <https://grouplens.org/datasets/movielens/>`_ Dataset
    """

    def __init__(self, ds_option, train=True):
        self.option = ds_option
        self.data_processor = MovielensProcessor(self.option)

        self.processed_path = os.path.join(ds_option.root_dir, ds_option.processed_folder, ds_option.ds_name)
        self.is_train_set = train
        self.data = {True: [], False: []}

    def __len__(self):
        return len(self.data[self.is_train_set])

    def __getitem__(self, item):
        return self.data[self.is_train_set].iloc[item].tolist()

    def load_data(self):
        if self.is_train_set:
            self.data[self.is_train_set] = pd.read_csv(os.path.join(self.processed_path, 'train.csv'))
        else:
            self.data[self.is_train_set] = pd.read_csv(os.path.join(self.processed_path, 'test.csv'))

    def download_and_process_data(self):
        self.data_processor.download(force=self.option.force_new)
        self.data_processor.map_dataset(force=self.option.force_new)
        self.data_processor.split_train_test_dataset(force=self.option.force_new)
