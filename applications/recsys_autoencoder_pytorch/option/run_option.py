import os
import json
import shutil
import logging
from .logger import FileLogger


class RunOption(dict):
    _default_attrs = {
        'g_root_dir': 'root_dir',
        'g_save_dir': 'save_dir',
        'g_file_name': 'option.json',
        'g_force_new_option': False,  # force new option, override current option file
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._set_key_value_pairs(self._default_attrs)
        self._read_kwargs(**kwargs)
        self._create_save_dir()
        self._create_or_load_option_file()

    def __getattr__(self, item):
        return self[item]

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, item):
        del self[item]

    def __dir__(self):
        return super().__dir__() + [str(k) for k in self.keys()]

    def _set_key_value_pairs(self, dict):
        for key, value in dict.items():
            self[key] = value

    def _read_kwargs(self, **kwargs):
        for key, value in kwargs.items():
            self[key] = value

    def _create_save_dir(self):
        save_dir_path = self.get_working_dir()
        os.makedirs(save_dir_path, exist_ok=True)

    def _create_or_load_option_file(self):
        file_path = os.path.join(self.get_working_dir(), self.g_file_name)

        if self.g_force_new_option or not os.path.exists(file_path):
            self.save(file_path)
        else:
            with open(file_path, 'r') as f:
                load_option = json.load(f)
                self._set_key_value_pairs(load_option)

    def save(self, file_path=None):
        if file_path is None:
            file_path = os.path.join(self.get_working_dir(), self.g_file_name)

        with open(file_path, 'w') as f:
            json.dump(self, f, indent=True)

    def deleteOption(self):
        shutil.rmtree(self.root_dir)

    def logger(self, level=logging.DEBUG):
        file_path = os.path.join(self.get_working_dir(), 'log.txt')
        return FileLogger(file_path, level)

    def get_working_dir(self):
        return os.path.join(self.g_root_dir, self.g_save_dir)
