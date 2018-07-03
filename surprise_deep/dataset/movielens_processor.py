import os
import errno
import pandas as pd
from six.moves import urllib


class MovielensProcessor():
    """`MovieLens <https://grouplens.org/datasets/movielens/>`_ Dataset

    """
    _ds_names = ['100k', '1m', '10m', '20m', '26m', 'serendipity']

    _urls = {
        _ds_names[0]: {
            'file': 'http://files.grouplens.org/datasets/movielens/ml-latest-small.zip',
            'year': 2016,
            'delimiter': ',',
            'rating_file': 'ratings.csv'
        },
        _ds_names[1]: {
            'file': 'http://files.grouplens.org/datasets/movielens/ml-1m.zip',
            'year': 2003,
            'delimiter': '::',
            'rating_file': 'ratings.dat'
        },
        _ds_names[2]: {
            'file': 'http://files.grouplens.org/datasets/movielens/ml-10m.zip',
            'year': 2009,
            'delimiter': '::',
            'rating_file': 'ratings.dat'
        },
        _ds_names[3]: {
            'file': 'http://files.grouplens.org/datasets/movielens/ml-20m.zip',
            'year': 2016,
            'delimiter': ',',
            'rating_file': 'ratings.csv'
        },
        _ds_names[4]: {
            'file': 'http://files.grouplens.org/datasets/movielens/ml-latest.zip',
            'year': 2017,
            'youtube': 'http://files.grouplens.org/datasets/movielens/ml-20m-youtube.zip',
            'delimiter': ',',
            'rating_file': 'ratings.csv'
        },
        _ds_names[5]: {
            'file': 'http://files.grouplens.org/datasets/serendipity-sac2018/serendipity-sac2018.zip',
            'year': 2018,

        }
    }

    _raw_folder = 'raw'
    _mapping_folder = 'map'
    _processed_folder = 'processed'

    def __init__(self, ds_option):
        self.option = ds_option
        self.root = self.option.root_dir
        self.ds_name = self.option.ds_name

        url_dict = self._urls[self.ds_name]
        self.url = url_dict['file']
        self.delimiter = url_dict['delimiter']
        self.rating_file_name = url_dict['rating_file']
        self.filename_zip = self.url.rpartition('/')[2]
        self.filename = self.filename_zip.replace('.zip', '')
        self.file_path_zip = os.path.join(self.root, self._raw_folder, self.ds_name, self.filename_zip)
        self.ds_folder = os.path.join(self.root, self._raw_folder, self.ds_name)

    def download(self, force=False):
        self._create_dataset_dir(self._raw_folder, self.ds_name)
        if self._check_exists(self.file_path_zip) and not force:
            print('file: ' + self.filename_zip + ' existed, skip download')
        else:
            print('Downloading ' + self.url)
            data = urllib.request.urlopen(self.url)
            with open(self.file_path_zip, 'wb') as f:
                f.write(data.read())

        self._unzip_file(self.file_path_zip, self.ds_name)

    # todo: later optimize procesing data with saving checkpoint
    def map_dataset(self, force=False):
        done_file = os.path.join(self.root, self._mapping_folder, self.ds_name, 'done')
        if os.path.exists(done_file) and not force:
            print(f'Already mapped dataset {self.ds_name}, skip.')
            return

        self._create_dataset_dir(self._mapping_folder, self.ds_name)
        raw_rating_file = os.path.join(self.ds_folder, self.filename, self.rating_file_name)
        user_id_map_file = os.path.join(self.root, self._mapping_folder, self.ds_name, 'map_user_id.csv')
        movie_id_map_file = os.path.join(self.root, self._mapping_folder, self.ds_name, 'map_movie_id.csv')
        map_rating_file = os.path.join(self.root, self._mapping_folder, self.ds_name, 'map_rating.csv')

        map_user_map = {}
        map_user_str = ''
        map_movie_map = {}
        map_movie_str = ''
        map_rating_str = ''

        user_count = 0
        movie_count = 0
        rating_count = 0
        timestamp_count = 0
        with open(raw_rating_file, 'r') as raw_f, \
                open(user_id_map_file, 'w') as user_f, \
                open(movie_id_map_file, 'w') as movie_f, \
                open(map_rating_file, 'w') as rating_f, \
                open(done_file, 'w') as done_f:

            for line in raw_f:
                try:
                    items = line.split(self.delimiter)
                    raw_user_id = int(items[0])
                    raw_movie_id = int(items[1])
                    raw_rating = float(items[2])
                    raw_time = int(items[3])
                    temp_movie_id = movie_count
                    temp_user_id = user_count
                    if raw_user_id not in map_user_map:
                        map_user_map[raw_user_id] = user_count
                        temp_user_id = user_count
                        user_count += 1
                    else:
                        temp_user_id = map_user_map[raw_user_id]

                    if raw_movie_id not in map_movie_map:
                        map_movie_map[raw_movie_id] = movie_count
                        temp_movie_id = movie_count
                        movie_count += 1
                    else:
                        temp_movie_id = map_movie_map[raw_movie_id]

                    map_rating_str += f'{temp_user_id},{temp_movie_id},{raw_rating},{raw_time}\n'
                    rating_count += 1
                    timestamp_count += 1
                    print('mapping rating ', rating_count)

                except Exception as e:
                    continue

            user_f.write(map_user_str)
            movie_f.write(map_movie_str)
            rating_f.write(map_rating_str)
            self.option.rating_columns = ['userId', 'movieId', 'rating', 'timestamp']
            self.option.rating_columns_unique_count = [user_count, movie_count, rating_count, timestamp_count]
            self.option.pivot_indexes = [0, 1]

            self.option.save()
            done_f.write("done")

    def split_train_test_dataset(self, force=False):
        self._create_dataset_dir(self._processed_folder, self.ds_name)
        done_file = os.path.join(self.root, self._processed_folder, self.ds_name, 'done')
        if os.path.exists(done_file) and not force:
            print(f'Already processed dataset {self.ds_name}, skip.')
            return

        print(f'Processed dataset {self.ds_name}...')
        map_rating_file = os.path.join(self.root, self._mapping_folder, self.ds_name, 'map_rating.csv')
        train_file = os.path.join(self.root, self._processed_folder, self.ds_name, 'train.csv')
        test_file = os.path.join(self.root, self._processed_folder, self.ds_name, 'test.csv')
        df = pd.read_csv(map_rating_file, names=['userId', 'movieId', 'rating', 'timestamp'])
        split_index = int(len(df) * 0.7)
        train_ds = df[0:split_index]
        test_ds = df[split_index:len(df)]
        train_ds.to_csv(train_file, header=True, index=False)
        test_ds.to_csv(test_file, header=True, index=False)
        with open(done_file, 'w') as f:
            f.write('done')

    def _check_exists(self, file_path):
        return os.path.exists(os.path.realpath(file_path))

    def _unzip_file(self, file_path, ds_folder):
        extract_folder = os.path.join(self.root, self._raw_folder, self.ds_name)
        if os.path.exists(os.path.join(ds_folder, self.filename)):
            return

        import zipfile
        with zipfile.ZipFile(file_path) as out_f:
            print('Unzipping ' + file_path)
            out_f.extractall(extract_folder)

    def _create_dataset_dir(self, dir, ds_name):
        try:
            os.makedirs(os.path.join(self.root, dir, ds_name))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise
