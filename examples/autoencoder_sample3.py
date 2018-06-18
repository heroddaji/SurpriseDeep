import sys

from recommender_systems import Autoencoder

sys.path.append('..')

from surprise_deep import *

ml_100k = MOVIELENS('movielens_ds', ds_name='100k', download=True, process_data=True)
ml_1m = MOVIELENS('movielens_ds', ds_name='1m', download=True, process_data=True)
# ml_10m = MOVIELENS('movielens_ds', ds_name='10m', download=True, process_data=True)
ml_20m = MOVIELENS('movielens_ds', ds_name='20m', download=True, process_data=True)
ml_26m = MOVIELENS('movielens_ds', ds_name='26m', download=True, process_data=True)




# bug: always unzip 10m file
