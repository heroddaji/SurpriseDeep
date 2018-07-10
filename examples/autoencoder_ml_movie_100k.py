import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append('../..')
sys.path.append('..')
print(sys.path)

from surprise_deep import *

pre_train = True

if not pre_train:
    ml_option_100k = DatasetOption(root_dir='ds_movielens', ds_name='100k', save_dir='100k_items')
    ml_option_100k.pivot_indexes = [1, 0]
    ml_option_100k.save()
    ml_ds_train_100k = Movielens(ml_option_100k)
    ml_ds_train_100k.download_and_process_data()
    ml_ds_test_100k = Movielens(ml_option_100k, train=False)

    model_option_100k = ModelOption(root_dir="recsys_deeplearning", save_dir="autoencoder_100k_items")
    model_option_100k.activation = 'selu'
    model_option_100k.save()
    model_100k = Autoencoder(model_option_100k,
                             input_dim=ml_option_100k.rating_columns_unique_count[
                                 ml_option_100k.pivot_indexes[1]])  # get the movie count as number of columns
    print(model_100k)
    model_100k.learn(ml_ds_train_100k)
    model_100k.save_model('autoencoder_100k.model')
    model_100k.load_model('autoencoder_100k.model')
    model_100k.evaluate(ml_ds_test_100k, 'movielens_100k_preds.txt')
    model_100k.cal_RMSE("movielens_100k_preds.txt")
else:
    ml_option_100k = DatasetOption(root_dir='ds_movielens', ds_name='100k', save_dir='100k_items')
    ml_ds_test_100k = Movielens(ml_option_100k, train=False)

    model_option_100k = ModelOption(root_dir="recsys_deeplearning", save_dir="autoencoder_100k_items")
    model_option_100k.activation = 'selu'
    model_option_100k.save()
    model_100k = Autoencoder(model_option_100k,
                             input_dim=ml_option_100k.rating_columns_unique_count[
                                 ml_option_100k.pivot_indexes[1]])  # get the movie count as number of columns

    model_100k.load_model('autoencoder_100k.model')
    model_100k.evaluate(ml_ds_test_100k, 'movielens_100k_preds.txt')
    model_100k.cal_RMSE("movielens_100k_preds.txt")
# bug: always unzip 10m file
