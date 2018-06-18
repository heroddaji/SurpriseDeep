import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from recommender_systems import Autoencoder, MovielensProcessor, Movielens

sys.path.append('..')
from surprise_deep import *

ml_processor = MovielensProcessor('movielens', ds_name='100k')
ml_processor.download()
ml_processor.map_dataset()
ml_processor.split_train_test_dataset()
ml_ds = Movielens('movielens', ds_name='100k')

batch_size = 256
column_size = 9066
use_gpu = torch.cuda.is_available()  # global flag

train_loader = DataLoader(ml_ds, batch_size=batch_size)

model = Autoencoder(column_size, 30)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for i, batch_items in enumerate(train_loader):
    index_tensors = torch.tensor([batch_items[:, 0],batch_items[:, 1]])
    value_tensors = torch.tensor(batch_items[:, 2])
    sparse_rating_matrix = torch.sparse.FloatTensor(index_tensors, value_tensors, torch.Size(batch_size, column_size))
    dense_rating_matrix = sparse_rating_matrix.cuda().to_dense() if use_gpu else sparse_rating_matrix.to_dense()
    optimizer.zero_grad()
    outputs = model(dense_rating_matrix)

algo = Autoencoder(in_dim=600, hid_dim=3)

# bug: always unzip 10m file
