import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append('../..')
sys.path.append('..')
print(sys.path)

from surprise_deep import *

param_options = {
    'activation': ['selu', 'relu', 'relu6', 'adam', 'momentum', 'sigmoid', 'tanh'],
    'hidden_layers': [2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2],
    'optimizer': ['adam', 'adagrad', 'momentum', 'rmsprop', 'sgd'],
    'learning_rate': [0.0001, 0.01],
    'drop_prob': [0, 0.9],
    'noise_prob': [0, 0.9],
    'test_masking_rate': [0.3, 0.9],
    'decoder_constraint': [True, False],
    'normalize_data': [True, False],
    'test_split_rate': [0.1, 0.4],
    'pivot_indexes': [[0, 1], [0, 1]]
}
