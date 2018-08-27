import sys

sys.path.append('../..')
sys.path.append('..')
print(sys.path)

import os
import pandas as pd
from surprise_deep import *
from py2neo import Graph
from py2neo import Node

graph = Graph("bolt://128.199.71.16:7687", auth=("neo4j", "DaiNeo4j123!@#"))
ds_ml_opt = None



def download_ml(ds_name='100k'):
    global ds_ml_opt
    ds_ml_opt = DatasetOption(g_root_dir='ds', g_save_dir=ds_name,d_ds_name=ds_name)
    ds_ml = Movielens(ds_ml_opt)
    # ds_ml.download_and_process_data()


def insert_ml_movies(unzip_folder, csv_file, graph_type):
    raw_movies_db_path = os.path.join(ds_ml_opt.get_working_dir(), 'raw', unzip_folder, csv_file)
    df_raw_movies = pd.read_csv(raw_movies_db_path)
    print(df_raw_movies.head())
    insert_dataframe(df_raw_movies, graph_type)


def insert_ml_users(max_id, graph_type):
    insert_list([i for i in range(1, max_id)], graph_type)


def insert_ml_ratings(unzip_folder, csv_file, graph_type):
    raw_ratings_db_path = os.path.join(ds_ml_opt.get_working_dir(), 'raw', unzip_folder, csv_file)
    df_raw_ratings = pd.read_csv(raw_ratings_db_path)
    insert_dataframe(df_raw_ratings,graph_type)


def insert_ml_tags(unzip_folder, csv_file, graph_type):
    raw_tag_db_path = os.path.join(ds_ml_opt.get_working_dir(), 'raw', unzip_folder, csv_file)
    df_raw_tag = pd.read_csv(raw_tag_db_path)
    print(df_raw_tag.head())
    insert_dataframe(df_raw_tag, graph_type)


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

def handle_100k_ds():
    download_ml('100k')
    insert_ml_movies(unzip_folder='ml-latest-small', csv_file='movies.csv', graph_type='ml100k_movie')
    insert_ml_users(max_id=270900, graph_type='ml100k_user')
    insert_ml_tags(unzip_folder='ml-latest-small', csv_file='tags.csv', graph_type='ml100k_tag')
    insert_ml_ratings(unzip_folder='ml-latest-small', csv_file='ratings.csv', graph_type='ml100k_rating')


def handle_26m_ds():
    download_ml('26m')
    insert_ml_movies(unzip_folder='ml-latest', csv_file='movies.csv', graph_type='ml26m_movie')
    insert_ml_users(max_id=270900, graph_type='ml26m_user')
    insert_ml_tags(unzip_folder='ml-latest', csv_file='tags.csv', graph_type='ml26m_tag')
    insert_ml_ratings(unzip_folder='ml-latest', csv_file='ratings.csv', graph_type='ml26m_rating')

if __name__ == '__main__':
    handle_26m_ds()


'''
LOAD CSV WITH HEADERS FROM 'file:///ml_26m/ratings.csv' AS line
CREATE (:ml26m_rating {userId:toInteger(line.userId), movieId:toInteger(line.movieId), rating:line.rating, timestamp:line.timestamp})
'''