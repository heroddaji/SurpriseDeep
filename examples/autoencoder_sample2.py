import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append('../..')
sys.path.append('..')
print(sys.path)

from surprise_deep import *
from recommender_systems import Autoencoder, MovielensProcessor, Movielens

ml_option = DatasetOption(root_dir='ds_movielens', ds_name='100k')
ml_ds_train = Movielens(ml_option)
ml_ds_train.download_and_process_data()
ml_ds_test = Movielens(ml_option, train=False)

model_option = ModelOption(root_dir="ml_autoencoder")
model = Autoencoder(model_option, input_dim=ml_option.rating_columns_unique_count[
    ml_option.pivot_indexes[1]])  # get the movie count as number of columns
print(model)
model.fit(ml_ds_train)

"""
batch_size = 5012
column_size = 9066
num_epochs = 10
use_gpu = torch.cuda.is_available()  # global flag

train_loader = DataLoader(ml_ds, batch_size=batch_size)

model = Autoencoder(column_size, 100)
print(model)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(num_epochs):
    t0 = time.time()
    model.train()
    for i, batch_items in enumerate(train_loader):
        try:
            index_tensors = torch.LongTensor([batch_items[0].tolist(), batch_items[1].tolist()])
            value_tensors = torch.FloatTensor(batch_items[2].tolist())
            sparse_rating_matrix = torch.sparse.FloatTensor(index_tensors, value_tensors,
                                                            torch.Size([batch_size, column_size]))
            dense_rating_matrix = sparse_rating_matrix.cuda().to_dense() if use_gpu else sparse_rating_matrix.to_dense()
            outputs = model(dense_rating_matrix)

            # mask = dense_rating_matrix != 0
            # num_ratings = torch.sum(mask.float())

            loss = criterion(outputs, dense_rating_matrix)
            # loss = loss / num_ratings

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f Time: %.2fs' % (
                epoch + 1, num_epochs, i + 1, len(ml_ds) // batch_size, loss.data[0], time.time() - t0))
        except Exception as e:
            print(e)

# bug: always unzip 10m file
"""
