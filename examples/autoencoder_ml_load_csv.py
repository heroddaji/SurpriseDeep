import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import random
import json
import pandas as pd
import numpy as np

sys.path.append('../..')
sys.path.append('..')
print(sys.path)

from surprise_deep import *

dataset='100k'
model_name = 'autoencoder_100k'
predict_name = 'movielens_100k_preds.txt'

def read_params_csv(file_path):
    df = pd.read_csv(file_path)
    return df

if __name__ == '__main__':
    df = read_params_csv(r'C:\Users\45027285\Google Drive\study_coding\lab_papers\flexEncoder_paper\flex_encoder_paper_python_notebook\records.csv')
    ds_option = DatasetOption(g_root_dir='ds_movielens', g_save_dir=dataset, d_ds_name=dataset)
    model_option = ModelOption(g_root_dir="recsys_deeplearning", g_save_dir=model_name)

    #run the interested  row
    row = df.iloc[0,:]
    for name, value in row.items():
        if ds_option.get(name, None) !=None:
            ds_option[name] = value
            if name == 'dp_pivot_indexes':
                if value.startswith('['):
                    value = value[1:-1]
                    value = value.split(',')
                    value = [int(i) for i in value]
                    a=0
                else:
                    value = [int(value)]
            ds_option[name] = value

        if model_option.get(name, None) !=None:
            model_option[name] = value
            if name == 'mp_hidden_layers':
                if value.startswith('['):
                    value = value[1:-1]
                    value = value.split(',')
                    value = [int(i) for i in value]
                    a=0
                else:
                    value = [int(value)]

                model_option[name] = value

    for key, value in ds_option.copy().items():
        if isinstance(value, np.bool_):
            ds_option[key] = bool(value)

    for key, value in model_option.copy().items():
        if isinstance(value, np.bool_):
            model_option[key] = bool(value)


    ml_ds_train = Movielens(ds_option)
    ml_ds_train.download_and_process_data()
    ml_ds_test = Movielens(ds_option, train=False)
    model = Autoencoder(model_option,
                        input_dim=ds_option.d_rating_columns_unique_count[
                            ds_option.dp_pivot_indexes[1]])  # get the movie count as number of columns
    print(model)
    model.learn(ml_ds_train)
    model.save_model(f'{model_name}.model')
    model.load_model(f'{predict_name}.model')
    model.evaluate(ml_ds_test, f'{predict_name}')
    model.cal_RMSE(f'{predict_name}')
