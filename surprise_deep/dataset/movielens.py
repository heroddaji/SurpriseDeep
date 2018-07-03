import os
import torch
import torch.utils.data as data
import pandas as pd
from .movielens_processor import MovielensProcessor


class Movielens(data.Dataset):
    """`MovieLens <https://grouplens.org/datasets/movielens/>`_ Dataset
    """

    def __init__(self, ds_option, train=True):
        self.option = ds_option
        self.data_processor = MovielensProcessor(self.option)

        self.processed_path = os.path.join(ds_option.root_dir, ds_option.processed_folder, ds_option.ds_name)
        self.is_train_set = train
        self.data = {True: [], False: []}
        self.group_data = None

    def __len__(self):
        return len(self.data[self.is_train_set])

    def __getitem__(self, item):
        return self.data[self.is_train_set].iloc[item].tolist()

    def _load_data(self):
        if self.is_train_set:
            self.data[self.is_train_set] = pd.read_csv(os.path.join(self.processed_path, 'train.csv'))
        else:
            self.data[self.is_train_set] = pd.read_csv(os.path.join(self.processed_path, 'test.csv'))

        pivot_indexes = self.option.pivot_indexes
        group_row_key = self.option.rating_columns[pivot_indexes[0]]
        self.group_data = self.data[self.is_train_set].groupby(group_row_key)

    def download_and_process_data(self):
        self.data_processor.download(force=self.option.force_new)
        self.data_processor.map_dataset(force=self.option.force_new)
        self.data_processor.split_train_test_dataset(force=self.option.force_new)

    def get_mini_batch(self, input_dim, batch_size=1, ):
        if not isinstance(self.data[self.is_train_set], pd.DataFrame):
            self._load_data()

        index1 = []
        index2 = []
        rating = []
        for index, group in self.group_data:
            index1 += group.iloc[:, 0].tolist()
            index2 += group.iloc[:, 1].tolist()
            rating += group.iloc[:, 2].tolist()
            if (index + 1) % batch_size == 0 or index == (len(self.group_data) - 1):
                i_torch = torch.LongTensor([index1, index2])
                v_torch = torch.FloatTensor(rating)
                mini_batch = torch.sparse.FloatTensor(i_torch, v_torch, torch.Size([max(index1) + 1, input_dim]))
                yield mini_batch
