import os
import torch
import torch.utils.data as data
import pandas as pd


class Movielens(data.Dataset):
    """`MovieLens <https://grouplens.org/datasets/movielens/>`_ Dataset

    """
    _ds_names = ['100k', '1m', '10m', '20m', '26m', 'serendipity']
    _raw_folder = 'raw'
    _processed_folder = 'processed'

    def __init__(self, root, ds_name, train=True):
        self.root = 'ds_' + root
        self.ds_name = ds_name
        self.processed_path = os.path.join(self.root, self._processed_folder, self.ds_name)
        self.is_train_set = train
        self.data = {True:[], False:[]}

        if self.is_train_set:
            self.data[self.is_train_set] = pd.read_csv(os.path.join(self.processed_path, 'train.csv'))
        else:
            self.data[self.is_train_set] = pd.read_csv(os.path.join(self.processed_path, 'test.csv'))

    def __len__(self):
        return len(self.data[self.is_train_set])

    def __getitem__(self, item):
        return self.data[self.is_train_set].iloc[item].as_matrix()
