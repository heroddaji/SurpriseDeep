from .run_option import RunOption
import os
import json


class DatasetOption(RunOption):
    _default_attrs = {
        'root_dir': 'dataset',
        'save_dir': 'save_dir',
        'file_name': 'dataset_option.json',
        'force_new': False,  # force create new option file
        'force_download': False,
        'force_map': False,
        'force_split': True,
        'ds_name': '100k',
        'raw_folder': 'raw',
        'processed_folder': 'processed',
        'map_folder': 'map',
        'rating_columns': ['userId', 'movieId', 'rating', 'timestamp'],  # run for user of movie
        'pivot_indexes': [
            0,
            1
        ],
        'rating_columns_unique_count': [0, 0, 0, 0],
        'test_split_rate': 0.3,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
