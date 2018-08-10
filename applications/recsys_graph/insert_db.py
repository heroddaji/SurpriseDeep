import sys

sys.path.append('../..')
sys.path.append('..')
print(sys.path)

import os
import json
import pandas as pd
from surprise_deep import *

from py2neo import Graph
from py2neo import Node

graph = Graph("bolt://128.199.71.16:7687", auth=("neo4j", "DaiNeo4j123!@#"))
ds_ml100k_opt = None


def download_ml100k():
    global ds_ml100k_opt
    ds_ml100k_opt = DatasetOption(g_root_dir='ds', g_save_dir='100k')
    ds_ml100k = Movielens(ds_ml100k_opt)
    ds_ml100k.download_and_process_data()


def insert_ml100k_movies():
    raw_movies_db_path = os.path.join(ds_ml100k_opt.get_working_dir(), 'raw', 'ml-latest-small', 'movies.csv')
    df_raw_movies = pd.read_csv(raw_movies_db_path)
    print(df_raw_movies.head())
    insert_dataframe(df_raw_movies, "ml100k_movie")


def insert_ml100k_users():
    insert_list([i for i in range(1, 672)], "ml100k_user")


def insert_ml100k_ratings():
    raw_ratings_db_path = os.path.join(ds_ml100k_opt.get_working_dir(), 'raw', 'ml-latest-small', 'ratings.csv')
    df_raw_ratings = pd.read_csv(raw_ratings_db_path)
    insert_dataframe(df_raw_ratings, "ml100k_rating")


def insert_ml100k_tags():
    raw_movies_db_path = os.path.join(ds_ml100k_opt.get_working_dir(), 'raw', 'ml-latest-small', 'movies.csv')
    df_raw_movies = pd.read_csv(raw_movies_db_path)
    print(df_raw_movies.head())
    insert_dataframe(df_raw_movies, "ml100k_movie")


def insert_dataframe(df, type):
    for i, series in df.iterrows():
        node = Node(type)
        for key, value in series.items():
            node[key] = value
        graph.create(node)
        print("create node", node)


def insert_list(list, type):
    nodes = []
    for i in list:
        node = Node(type, id=i)
        nodes.append(node)
        graph.create(node)
        print(f'created node {node} for type:{type}')


if __name__ == '__main__':
    download_ml100k()
    # insert_ml100k_movies()
    # insert_ml100k_users()
    insert_ml100k_ratings()
    # insert_ml100k_tags()
