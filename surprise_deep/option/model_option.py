from .run_option import RunOption
import os
import json


class ModelOption(RunOption):
    _default_attrs = {
        'g_root_dir': 'root_dir',
        'g_save_dir': 'save_dir',
        'g_file_name': 'model_option.json',
        'g_force_new_option': False,  # force new option, override current option file
        'm_resume_training': False,
        'mp_learning_rate': 0.005,
        'mp_weight_decay': 0,
        'mp_drop_prob': 0.0,  # dropout drop probability
        'mp_noise_prob': 0.0,
        'mp_train_batch_size': 64,
        'mp_test_batch_size': 1,
        'mp_test_masking_rate': 0.5,
        'mp_num_epochs': 10,
        'mp_optimizer': 'adam',
        'mp_activation': 'selu',  # selu, relu6, etc
        'mp_hidden_layers': [128],
        'mp_aug_step': 1,
        'mp_aug_step_floor': False,
        'mp_decoder_constraint': False,  # reuse weight from the encoder if True
        'mp_normalize_data': True,
        'mp_last_layer_activations': True,
        'mp_prediction_floor': False,  # round off number  to near whole or half, e.g prediction is 1.65 -> 1.5
        'mp_loss_size_average': False  # when using MMSE loss, use size_average option
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
