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
        'rating_columns': ['userId', 'movieId', 'rating', 'timestamp'],  # run for user of movie
        "pivot_indexes": [
            0,
            1
        ],
        'rating_columns_unique_count': [1, 2, 3, 4],
        'shuffle': True,

    }
    _attrs = ['root_dir', 'file_name', 'force_new']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _read_kwargs(self, **kwargs):
        self.root_dir = kwargs.get(self._attrs[0], self._default_attrs[self._attrs[0]])
        self.file_name = kwargs.get(self._attrs[1], self._default_attrs[self._attrs[1]])
        self.force_new = kwargs.get(self._attrs[2], self._default_attrs[self._attrs[2]])
