from .run_option import RunOption


class DatasetOption(RunOption):
    _default_attrs = {
        'root_dir': 'dataset',
        'file_name': 'dataset_option.json',
        'force_new': False,
        'ds_name': '100k',
        'raw_folder': 'raw',
        'processed_folder': 'processed',
        'map_folder': 'map',
        'user_count':0,
        'movie_count':0,
        'rating_count':0,
    }
    _attrs = ['root_dir', 'file_name', 'force_new']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _read_kwargs(self, **kwargs):
        self.root_dir = kwargs.get(self._attrs[0], self._default_attrs[self._attrs[0]])
        self.file_name = kwargs.get(self._attrs[1], self._default_attrs[self._attrs[1]])
        self.force_new = kwargs.get(self._attrs[2], self._default_attrs[self._attrs[2]])
