import sys
# sys.path.append()
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import dnet

G = nx.MultiDiGraph()
D = dnet.HeteroGraph()
globalid = 1


def _read_users():
    # read users
    with open('../datasets/movielens/ml-100k/u.user', 'r') as f:
        line = f.readline()
        while line is not None:
            if line == '':
                break
            line = line[:-1]
            user_data = line.split('|')
            # G.add_node(int(user_data[0]),
            #            age=user_data[1],
            #            gender=user_data[2],
            #            occupation=user_data[3],
            #            zip=user_data[4],
            #            type='user')
            D.add_node(node_id=int(user_data[0]),
                       type='user',
                       age=user_data[1],
                       gender=user_data[2],
                       occupation=user_data[3],
                       zip=user_data[4])
            print('added user:', user_data[0])
            line = f.readline()


def _read_movies():
    with open('../datasets/movielens/ml-100k/u.item', 'r', encoding="ISO-8859-1") as f:
        line = f.readline()
        while line is not None:
            if line == '':
                break
            line = line[:-1]
            movie_data = line.split('|')
            # G.add_node(int(movie_data[0]),
            #            title=movie_data[1],
            #            releaseDate=movie_data[2],
            #            videoReleaseDate=movie_data[3],
            #            imdbUrl=movie_data[4],
            #            genre=[int(movie_data[5]),
            #                   int(movie_data[6]),
            #                   int(movie_data[7]),
            #                   int(movie_data[8]),
            #                   int(movie_data[9]),
            #                   int(movie_data[10]),
            #                   int(movie_data[11]),
            #                   int(movie_data[12]),
            #                   int(movie_data[13]),
            #                   int(movie_data[14]),
            #                   int(movie_data[15]),
            #                   int(movie_data[16]),
            #                   int(movie_data[17]),
            #                   int(movie_data[18]),
            #                   int(movie_data[19]),
            #                   int(movie_data[20]),
            #                   int(movie_data[21]),
            #                   int(movie_data[22]),
            #                   int(movie_data[23]),
            #                   ],
            #            type='movie'),
            D.add_node(node_id=int(movie_data[0]),
                       type='movie',
                       title=movie_data[1],
                       releaseDate=movie_data[2],
                       videoReleaseDate=movie_data[3],
                       imdbUrl=movie_data[4],
                       genre=[int(movie_data[5]),
                              int(movie_data[6]),
                              int(movie_data[7]),
                              int(movie_data[8]),
                              int(movie_data[9]),
                              int(movie_data[10]),
                              int(movie_data[11]),
                              int(movie_data[12]),
                              int(movie_data[13]),
                              int(movie_data[14]),
                              int(movie_data[15]),
                              int(movie_data[16]),
                              int(movie_data[17]),
                              int(movie_data[18]),
                              int(movie_data[19]),
                              int(movie_data[20]),
                              int(movie_data[21]),
                              int(movie_data[22]),
                              int(movie_data[23]),
                              ]),
            print('added movie:', movie_data[0])
            line = f.readline()
            '''
            movie id 0| movie title 1| release date 2| video release date 3|
              IMDb URL 4| unknown 5| Action 6| Adventure7 | Animation 8|
              Children's 9| Comedy 10| Crime 11| Documentary 12| Drama 13| Fantasy 14|
              Film-Noir 15| Horror 16| Musical 17| Mystery 18| Romance 19| Sci-Fi 20|
              Thriller 21| War 22| Western 23|
            '''


def _read_ratings():
    # read rating file, add relationship
    with open('../datasets/movielens/ml-100k/u.data', 'r', encoding="ISO-8859-1") as f:
        line = f.readline()
        while line is not None:
            if line == '':
                break
            line = line[:-1]
            rating_data = line.split('\t')
            user = int(rating_data[0])
            movie = int(rating_data[1])
            rating = int(rating_data[2])
            timestamp = int(rating_data[3])
            # G.add_edge(user, movie, weight=rating, timestamp=timestamp)
            D.add_edge(user, 'user', movie, 'movie', weight=rating, timestamp=timestamp)
            print('added rating:', user, movie, rating)
            line = f.readline()


def _save_graph_format():
    nx.write_gml(G, '../datasets/movielens/movielens_100k.gml')


def _load_graph():
    global G
    G = nx.read_gml('../datasets/movielens/movielens_100k.gml')
    # nx.draw(G)
    # plt.show()


def form_hete_graph():
    _read_users()
    _read_movies()
    _read_ratings()
    # _save_graph_format()
    # _load_graph()


def experiment_walk_rank():
    users = D.nodes['user']
    print(users)


if __name__ == '__main__':
    # form_hete_graph()
    # experiment_walk_rank()
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver("bolt://localhost:17687", auth=("neo4j", "DaiNeo4j123!@#"))


    def add_friend(tx, name, friend_name):
        tx.run("MERGE (a:Person {name: $name}) "
               "MERGE (a)-[:KNOWS]->(friend:Person {name: $friend_name})",
               name=name, friend_name=friend_name)


    def print_friends(tx, name):
        for record in tx.run("MATCH (a:Person)-[:KNOWS]->(friend) WHERE a.name = $name "
                             "RETURN friend.name ORDER BY friend.name", name=name):
            print(record["friend.name"])


    with driver.session() as session:
        session.write_transaction(add_friend, "Arthur", "Guinevere")
        session.write_transaction(add_friend, "Arthur", "Lancelot")
        session.write_transaction(add_friend, "Arthur", "Merlin")
        session.read_transaction(print_friends, "Arthur")

