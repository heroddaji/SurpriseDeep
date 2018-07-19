from .run_option import RunOption
import os
import json


class DatasetOption(RunOption):
    _default_attrs = {
        'g_root_dir': 'root_dir',
        'g_save_dir': 'save_dir',
        'g_file_name': 'dataset_option.json',
        'g_force_new_option': False,
        'd_force_download': False,
        'd_force_map': False,
        'd_force_split': True,
        'd_ds_name': '100k',
        'd_raw_folder': 'raw',
        'd_processed_folder': 'processed',
        'd_map_folder': 'map',
        'd_rating_columns': ['userId', 'movieId', 'rating', 'timestamp'],  # run for user of movie
        'd_rating_columns_unique_count': [0, 0, 0, 0],
        'd_normalize_mapping': True,
        'dp_pivot_indexes': [
            0,
            1
        ],
        'dp_test_split_rate': 0.3,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
