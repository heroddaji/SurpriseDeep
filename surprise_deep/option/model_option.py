from .run_option import RunOption


class ModelOption(RunOption):
    _default_attrs = {
        "root_dir": "model",
        "file_name": "model_option.json",
        "force_new": False,
        "learning_rate": 0.0001,
        "weight_decay": 0,
        "drop_prob": 0.0,  # dropout drop probability
        "noise_prob": 0.0,
        "batch_size": 128,
        "num_epochs": 50,
        "optimizer": "adam",
        "activation": "relu",  # selu, relu6, etc
        "hidden_layers": [1024, 512, 512, 128],
        "decoder_constraint": True,  # ???
        "save_dir": "save_models"  # save and log directory inside the root_dir
    }
    _attrs = ["root_dir", "file_name"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _read_kwargs(self, **kwargs):
        self.root_dir = kwargs.get(self._attrs[0], self._default_attrs[self._attrs[0]])
        self.file_name = kwargs.get(self._attrs[1], self._default_attrs[self._attrs[1]])
