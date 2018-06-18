import sys
import torch
from torch.utils.data import DataLoader
from recommender_systems import Autoencoder, MovielensProcessor, Movielens

sys.path.append('..')
from surprise_deep import *

ml_processor = MovielensProcessor('movielens', ds_name='100k')
ml_processor.download()
ml_processor.map_dataset()
ml_processor.split_train_test_dataset()

ml_ds = Movielens('movielens',ds_name='100k')
train_loader = DataLoader(ml_ds, batch_size=256)

for i, batch_items in enumerate(train_loader):

    print(i,batch_items)

algo = Autoencoder(in_dim=600, hid_dim=3)

#bug: always unzip 10m file