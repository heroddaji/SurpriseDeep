from .run_option import RunOption
import os
import json


class ModelOption(RunOption):
    _default_attrs = {
        'root_dir': 'model',
        'save_dir': 'model',
        'file_name': 'model_option.json',
        'force_new': False,  # force new option
        'learning_rate': 0.005,
        'weight_decay': 0,
        'drop_prob': 0.0,  # dropout drop probability
        'noise_prob': 0.0,
        'train_batch_size': 64,
        'test_batch_size': 1,
        'test_masking_rate': 0.5,
        'num_epochs': 10,
        'optimizer': 'adam',
        'activation': 'selu',  # selu, relu6, etc
        'hidden_layers': [128],
        'aug_step': 1,
        'aug_step_floor': False,
        'decoder_constraint': False,  # reuse weight from the encoder if True
        'resume_training': True,
        'random_data_each_epoch': False,
        'normalize_data': True,
        'last_layer_activations': True,
        'prediction_floor': False  # round off number  to near whole or half, e.g prediction is 1.65 -> 1.5
    }
    _attrs = ['root_dir', 'save_dir', 'file_name']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
