import os
import torch
import torch.utils.data as data
import pandas as pd
from random import shuffle
import numpy as np
from .movielens_processor import MovielensProcessor


class Movielens(data.Dataset):
    """`MovieLens <https://grouplens.org/datasets/movielens/>`_ Dataset
    """

    def __init__(self, ds_option, train=True):
        self.option = ds_option
        self.data_processor = MovielensProcessor(self.option)

        self.processed_path = os.path.join(ds_option.get_working_dir(),
                                           ds_option.d_processed_folder,
                                           )
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

        pivot_indexes = self.option.dp_pivot_indexes
        group_row_key = self.option.d_rating_columns[pivot_indexes[0]]
        self.group_data = self.data[self.is_train_set].groupby(group_row_key)

    def download_and_process_data(self):
        self.data_processor.download()
        self.data_processor.map_dataset()
        if self.option.d_normalize_mapping:
            self.data_processor.normalize_user_data()
            self.data_processor.normalize_movie_data()
            self.data_processor.normalize_rating_data()
        self.data_processor.split_train_test_dataset()

    def get_mini_batch(self, input_dim, batch_size=1, test_masking_rate=0, is_normalize=False):
        if not isinstance(self.data[self.is_train_set], pd.DataFrame):
            self._load_data()

        sparse_row_index = []
        sparse_column_index = []
        sparse_value = []

        # shuffle to random group order
        random_groups = []
        for index, group in self.group_data:
            random_groups.append(group)
        shuffle(random_groups)
        pivot_indexes = self.option.dp_pivot_indexes
        for index, group in enumerate(random_groups):
            try:
                sparse_row_index += group.iloc[:, pivot_indexes[0]].astype(int).tolist()
                sparse_column_index += group.iloc[:, pivot_indexes[1]].astype(int).tolist()
                if is_normalize:
                    column = pivot_indexes[0] + 4 #todo: hardcode, bad, get the norm colume index       of either user or movie
                    sparse_value += group.iloc[:, column].astype(float).tolist()
                else:
                    sparse_value += group.iloc[:, 2].astype(float).tolist()

                # make certain % of input_values become zero
                if test_masking_rate > 0 and len(sparse_value) - 1 > 0:
                    random_mask_index = np.random.randint(0, len(sparse_value) - 1,
                                                          int(len(sparse_value) * test_masking_rate))
                if (index + 1) % batch_size == 0 or index == (len(random_groups) - 1):
                    i_torch = torch.LongTensor([sparse_row_index, sparse_column_index])
                    v_torch = torch.FloatTensor(sparse_value)
                    if test_masking_rate > 0 and len(sparse_value) - 1 > 0:
                        v_torch[random_mask_index] = 0
                    mini_batch = torch.sparse.FloatTensor(i_torch, v_torch,
                                                          torch.Size([int(max(sparse_row_index)) + 1, input_dim]))
                    yield sparse_row_index, sparse_column_index, sparse_value, mini_batch
                    sparse_row_index = []
                    sparse_column_index = []
                    sparse_value = []
            except Exception as e:
                continue  # skip any error when loading data into tensor
