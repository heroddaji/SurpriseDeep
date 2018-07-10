from .run_option import RunOption
import os
import json


class ModelOption(RunOption):
    _default_attrs = {
        'root_dir': 'model',
        'save_dir': 'model',
        'file_name': 'model_option.json',
        'force_new': False,  # force new option
        'learning_rate': 0.0001,
        'weight_decay': 0,
        'drop_prob': 0.0,  # dropout drop probability
        'noise_prob': 0.0,
        'train_batch_size': 128,
        'test_batch_size': 1,
        'test_masking_rate': 0.5,
        'num_epochs': 50,
        'optimizer': 'adam',
        'activation': 'relu',  # selu, relu6, etc
        'hidden_layers': [512, 256, 128],
        'decoder_constraint': True,  # ???
        'resume_training': True,
    }
    _attrs = ['root_dir', 'save_dir', 'file_name']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
