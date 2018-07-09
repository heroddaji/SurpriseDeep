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

ml_option = DatasetOption(root_dir='ds_movielens', ds_name='1m', save_dir='1m')
ml_ds_train = Movielens(ml_option)
ml_ds_train.download_and_process_data()
ml_ds_test = Movielens(ml_option, train=False)

model_option = ModelOption(root_dir="recsys_deeplearning", save_dir="autoencoder_selu_1m")
model_option.activation = 'selu'
model = Autoencoder(model_option,
                    input_dim=ml_option.rating_columns_unique_count[
                        ml_option.pivot_indexes[1]])  # get the movie count as number of columns
print(model)

model.learn(ml_ds_train)
model.save_model('autoencoder_selu_1m.model')
model.load_model('autoencoder_selu_1m.model')
model.evaluate(ml_ds_test, 'movielens_1m_preds.txt')
model.cal_RMSE("movielens_1m_preds.txt")

# bug: always unzip 10m file
