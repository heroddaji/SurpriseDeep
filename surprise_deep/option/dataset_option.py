from .run_option import RunOption
import os
import json


class DatasetOption(RunOption):
    _default_attrs = {
        'root_dir': 'dataset',
        'save_dir': 'save_dir',
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
    _attrs = ['root_dir', 'save_dir', 'file_name', 'ds_name']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

