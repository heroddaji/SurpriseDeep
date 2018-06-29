import os
import json
import shutil


class RunOption(dict):
    _default_attrs = {
        "root_dir": ".",
        "file_name": "option.json",
    }

    _attrs = ["root_dir", "file_name"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._set_key_value_pairs(self._default_attrs)
        self._read_kwargs(**kwargs)
        self._create_or_load_file()

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
        self.root_dir = kwargs.get(self._attrs[0], self._default_attrs[self._attrs[0]])

    def _create_or_load_file(self):
        os.makedirs(self.root_dir, exist_ok=True)
        file_path = os.path.join(self.root_dir, self.file_name)

        if self.force_new or not os.path.exists(file_path):
            self.save()
        else:
            with open(file_path, 'r') as f:
                load_option = json.load(f)
                self._set_key_value_pairs(load_option)

    def save(self):
        with open(os.path.join(self.root_dir, self.file_name), 'w') as f:
            json.dump(self, f, indent=True)

    def deleteOption(self):
        shutil.rmtree(self.root_dir)
