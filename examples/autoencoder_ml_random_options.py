import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import random

sys.path.append('../..')
sys.path.append('..')
print(sys.path)

from surprise_deep import *

param_options = {
    'train_batch_size': [1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1],
    'activation': ['selu', 'relu', 'relu6', 'elu', 'lrelu', 'sigmoid', 'tanh', 'swish'],
    'hidden_layers': [4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2],
    'optimizer': ['adam', 'adagrad', 'rmsprop', 'sgd'],  # momentum with scheduler??
    'learning_rate': [0.0001, 0.001, 0.005, 0.01, 0.1],
    'drop_prob': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'noise_prob': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'test_masking_rate': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'decoder_constraint': [True, False],
    'normalize_data': [True, False],
    'test_split_rate': [0.1, 0.2, 0.3, 0.4],
    'random_data_each_epoch': [True, False],
    'last_layer_activations': [True, False],
    'aug_step': [0, 1, 2, 3],
    # 'pivot_indexes': [[0, 1], [0, 1]],
    # other loss?
    #
    'RMSE': 0
}


def get_default_options():
    ds_option = DatasetOption(root_dir='ds_movielens', ds_name='100k', save_dir='100k')
    model_option = ModelOption(root_dir="recsys_deeplearning", save_dir="autoencoder_100k")
    return ds_option, model_option


def get_ds_model(ds_option, model_option):
    ds_train = Movielens(ds_option)
    ds_train.download_and_process_data()
    ds_test = Movielens(ds_option, train=False)

    model = Autoencoder(model_option,
                        input_dim=ds_option.rating_columns_unique_count[
                            ds_option.pivot_indexes[1]])
    return (ds_train, ds_test, model)


def write_msg(ds_option, model_option, rmse):
    msg = ''
    for key, values in param_options.items():
        if ds_option.get(key, None) != None:
            msg += str(ds_option[key]) + ','
        if model_option.get(key, None) != None:
            msg += str(model_option[key]) + ','

    msg += str(rmse)
    recorder.write_line(msg)


def execute(ds_option, model_option):
    (ds_train, ds_test, model) = get_ds_model(ds_option, model_option)
    model.learn(ds_train)
    model.evaluate(ds_test, 'movielens_100k_preds.txt')
    rmse = model.cal_RMSE("movielens_100k_preds.txt")
    return rmse


def run_single():
    for key, values in param_options.items():
        for value in values:
            rmses = []
            for i in range(2):
                (ds_option, model_option) = get_default_options()
                if key == 'hidden_layers':
                    model_option[key] = [value]
                else:
                    model_option[key] = value
                rmse = execute(ds_option, model_option)
                rmses.append(rmse)
            write_msg(ds_option, model_option, sum(rmses) / float(len(rmses)))

def run_random():
    pass

if __name__ == '__main__':
    recorder = Recorder(root_dir="recsys_deeplearning", save_dir="autoencoder_100k")
    recorder.open()
    column_names = ''
    for item in list(param_options.keys()):
        column_names += item + ','
    column_names = column_names[0:len(column_names) - 1]
    recorder.write_line(column_names)
    run_single()
    recorder.close()
