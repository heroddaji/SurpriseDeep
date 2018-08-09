import sys
sys.path.append('../..')
sys.path.append('..')
print(sys.path)

import os
import json
import pandas as pd
from surprise_deep import *

from db import neo4jdb

ds_ml100k_opt = DatasetOption()

def download_ml100k():
    global ds_ml100k_opt
    ds_ml100k_opt = DatasetOption(g_root_dir='ds',g_save_dir='100k')
    ds_ml100k = Movielens(ds_ml100k_opt)
    ds_ml100k.download_and_process_data()

def insert_ml100k():
    raw_movies_db_path = os.path.join(ds_ml100k_opt.get_working_dir(),'raw','ml-latest-small','movies.csv')
    df_raw_movies = pd.read_csv(raw_movies_db_path)
    print(df_raw_movies.head())

    with neo4jdb.get_session() as session:
        for i, row in df_raw_movies.iterrows():
            json_data = json.loads(row.to_json())
            json_data['genres'] = json_data['genres'].split("|")
            neo4jdb.insert_data(session, type='Movie', json_attrs=json.dumps(json_data))

if __name__ == '__main__':
    download_ml100k()
    insert_ml100k()
