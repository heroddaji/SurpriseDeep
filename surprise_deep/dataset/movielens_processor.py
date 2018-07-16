import os
import errno
import pandas as pd
from six.moves import urllib
from sklearn.model_selection import train_test_split
import shutil

mapping_item_name = 'map_movie_id.csv'
mapping_user_name = 'map_user_id.csv'
mapping_rating_name = 'map_rating.csv'
mapping_user_mean_name = 'map_user_mean.csv'
mapping_movie_mean_name = 'map_movie_mean.csv'
mapping_rating_norm_name = 'map_rating_norm.csv'


class MovielensProcessor():
    """`MovieLens <https://grouplens.org/datasets/movielens/>`_ Dataset

    """
    _ds_names = ['100k', '1m', '10m', '20m', '26m', 'serendipity', '100k_old']

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

        },
        _ds_names[6]: {
            'file': 'http://files.grouplens.org/datasets/movielens/ml-100k.zip',
            'year': 1998,
            'delimiter': '\t',
            'rating_file': 'u.data'
        }
    }

    _raw_folder = 'raw'
    _mapping_folder = 'map'
    _processed_folder = 'processed'

    def __init__(self, ds_option):
        self.option = ds_option
        self.logger = self.option.logger()
        self.root = self.option.root_dir
        self.save_dir = self.option.save_dir
        self.ds_name = self.option.ds_name

        url_dict = self._urls[self.ds_name]
        self.url = url_dict['file']
        self.delimiter = url_dict['delimiter']
        self.rating_file_name = url_dict['rating_file']
        self.filename_zip = self.url.rpartition('/')[2]
        self.filename = self.filename_zip.replace('.zip', '')
        self.file_path_zip = os.path.join(self.root, self.save_dir, self._raw_folder, self.filename_zip)
        self.ds_folder = os.path.join(self.root, self.save_dir, self._raw_folder)

    def download(self):
        force = self.option.force_download
        self._create_dataset_dir(self._raw_folder)
        if self._check_exists(self.file_path_zip) and not force:
            self.logger.debug('file: ' + self.filename_zip + ' existed, skip download')
        else:
            self.logger.debug('Downloading ' + self.url)
            data = urllib.request.urlopen(self.url)
            with open(self.file_path_zip, 'wb') as f:
                f.write(data.read())

        self._unzip_file(self.file_path_zip)

    def map_dataset(self):
        force = self.option.force_map
        done_file = os.path.join(self.root, self.save_dir, self._mapping_folder, 'done')
        if os.path.exists(done_file) and not force:
            self.logger.debug(f'Already mapped dataset {self.ds_name}, skip.')
            return

        self._create_dataset_dir(self._mapping_folder)
        raw_rating_file = os.path.join(self.ds_folder, self.filename, self.rating_file_name)
        user_id_map_file = os.path.join(self.root, self.save_dir, self._mapping_folder, mapping_user_name)
        movie_id_map_file = os.path.join(self.root, self.save_dir, self._mapping_folder,
                                         mapping_item_name)
        map_rating_file = os.path.join(self.root, self.save_dir, self._mapping_folder, mapping_rating_name)

        map_user_map = {}
        map_user_str = 'newId,originalId\n'
        map_movie_map = {}
        map_movie_str = 'newId,originalId\n'
        map_rating_str = 'userId,movieId,rating,timestamp\n'

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
                        map_user_str += f'{temp_user_id},{raw_user_id}\n'
                        user_count += 1
                    else:
                        temp_user_id = map_user_map[raw_user_id]

                    if raw_movie_id not in map_movie_map:
                        map_movie_map[raw_movie_id] = movie_count
                        temp_movie_id = movie_count
                        map_movie_str += f'{temp_movie_id},{raw_movie_id}\n'
                        movie_count += 1
                    else:
                        temp_movie_id = map_movie_map[raw_movie_id]

                    map_rating_str += f'{temp_user_id},{temp_movie_id},{raw_rating},{raw_time}\n'
                    rating_count += 1
                    timestamp_count += 1
                    self.logger.debug('mapping rating ', rating_count)

                except Exception as e:
                    continue

            user_f.write(map_user_str)
            movie_f.write(map_movie_str)
            rating_f.write(map_rating_str)
            self.option.rating_columns_unique_count = [user_count, movie_count, rating_count, timestamp_count]
            self.option.save()
            done_f.write("done")

    def split_train_test_dataset(self):
        force = self.option.force_split
        self._create_dataset_dir(self._processed_folder)
        done_file = os.path.join(self.root, self.save_dir, self._processed_folder, 'done')
        if os.path.exists(done_file) and not force:
            self.logger.debug(f'Already processed dataset {self.ds_name}, skip.')
            return

        self.logger.debug(f'Processed dataset {self.ds_name}...')
        map_rating_file = os.path.join(self.root, self.save_dir, self._mapping_folder, mapping_rating_name)
        train_file = os.path.join(self.root, self.save_dir, self._processed_folder, 'train.csv')
        test_file = os.path.join(self.root, self.save_dir, self._processed_folder, 'test.csv')
        if self.option.normalize_mapping:
            df = pd.read_csv(map_rating_file,
                             names=['userId', 'movieId', 'rating', 'timestamp', 'normUserRating', 'normMovieRating'])
        else:
            df = pd.read_csv(map_rating_file, names=['userId', 'movieId', 'rating', 'timestamp'])
        test_split_rate = self.option.test_split_rate
        train_ds, test_ds = train_test_split(df, test_size=test_split_rate)

        train_ds.to_csv(train_file, header=True, index=False)
        test_ds.to_csv(test_file, header=True, index=False)
        with open(done_file, 'w') as f:
            f.write('done')

    def normalize_user_data(self):
        map_rating_file = os.path.join(self.option.get_working_dir(), self._mapping_folder,
                                       mapping_rating_name)
        map_user_mean_file = os.path.join(self.option.get_working_dir(), self._mapping_folder,
                                          mapping_user_mean_name)
        if os.path.exists(map_user_mean_file):
            self.logger.debug(f'Already processed user mean {self.ds_name}, skip.')
            return

        df = pd.read_csv(map_rating_file)
        pivot_indexes = self.option.pivot_indexes
        group_user_key = self.option.rating_columns[pivot_indexes[0]]
        group_data = df.groupby(group_user_key)
        user_mean = 'userId,mean,count\n'
        for index, group in enumerate(group_data):
            user_df = group[1]
            user_id = group[0]
            user_mean += f"{user_id},{user_df['rating'].mean()},{user_df['rating'].count()}\n"
        with open(map_user_mean_file, 'w') as f:
            f.write(user_mean)

    def normalize_movie_data(self):
        map_rating_file = os.path.join(self.option.get_working_dir(), self._mapping_folder,
                                       mapping_rating_name)
        map_movie_mean_file = os.path.join(self.option.get_working_dir(), self._mapping_folder,
                                           mapping_movie_mean_name)

        if os.path.exists(map_movie_mean_file):
            self.logger.debug(f'Already processed movie mean {self.ds_name}, skip.')
            return

        df = pd.read_csv(map_rating_file)
        pivot_indexes = self.option.pivot_indexes
        group_movie_key = self.option.rating_columns[pivot_indexes[1]]
        group_data = df.groupby(group_movie_key)
        movie_mean = 'movieId,mean,count\n'
        for index, group in enumerate(group_data):
            movie_df = group[1]
            movie_id = group[0]
            movie_mean += f"{movie_id},{movie_df['rating'].mean()},{movie_df['rating'].count()}\n"
        with open(map_movie_mean_file, 'w') as f:
            f.write(movie_mean)

    def normalize_rating_data(self):
        map_movie_mean_file = os.path.join(self.option.get_working_dir(), self._mapping_folder,
                                           mapping_movie_mean_name)
        map_user_mean_file = os.path.join(self.option.get_working_dir(), self._mapping_folder,
                                          mapping_user_mean_name)
        map_rating_file = os.path.join(self.option.get_working_dir(), self._mapping_folder,
                                       mapping_rating_name)
        map_rating_norm_file = os.path.join(self.option.get_working_dir(), self._mapping_folder,
                                            mapping_rating_norm_name)
        if os.path.exists(map_rating_norm_file):
            self.logger.debug(f'Already processed rating mean {self.ds_name}, skip.')
            return
        else:
            self.logger.debug(f'Processing rating mean for dataset {self.ds_name} ')

        user_mean_df = pd.read_csv(map_user_mean_file)
        movie_mean_df = pd.read_csv(map_movie_mean_file)
        map_rating_df = pd.read_csv(map_rating_file)
        normalized_rating_content = 'userId,movieId,rating,timestamp,normUserRating,normMovieRating\n'

        # todo: add normalized rating and replace rating file, how to do it better?
        for index, row in map_rating_df.iterrows():
            userId = int(row['userId'])
            movieId = int(row['movieId'])
            rating = row['rating']
            timestamp = row['timestamp']
            user_mean = user_mean_df['mean'][userId]
            movie_mean = movie_mean_df['mean'][movieId]
            normalized_rating_content += f'{userId},{movieId},{rating},{timestamp},{rating-user_mean},{rating-movie_mean}\n'

        with open(map_rating_norm_file, 'w') as f:
            f.write(normalized_rating_content)

        os.remove(map_rating_file)
        shutil.copy2(map_rating_norm_file, map_rating_file)

    def _check_exists(self, file_path):
        return os.path.exists(file_path)

    def _unzip_file(self, file_path):
        extract_folder = os.path.join(self.root, self.save_dir, self._raw_folder)
        if os.path.exists(self.filename):
            return

        import zipfile
        extracted_name = ''
        with zipfile.ZipFile(file_path) as out_f:
            self.logger.debug('Unzipping ' + file_path)
            extracted = out_f.namelist()
            extracted_name = os.path.join(self.root, self.save_dir, self._raw_folder, extracted[0])
            out_f.extractall(extract_folder)
        os.rename(extracted_name, os.path.join(extract_folder,self.filename))

    def _create_dataset_dir(self, dir):
        try:
            os.makedirs(os.path.join(self.root, self.save_dir, dir))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise
