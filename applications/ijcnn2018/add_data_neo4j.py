from neo4j import GraphDatabase

users = []
movies = []
ratings = []
driver = GraphDatabase.driver("bolt://localhost:17687", auth=("neo4j", "DaiNeo4j123!@#"))


def add_movielens_data(tx):
    # _add_movies(tx)
    # _add_users(tx)
    _add_ratings(tx)


def _add_users(tx):
    with open('../../datasets/movielens/ml-100k/u.user', 'r') as f:
        line = f.readline()
        while line is not None:
            if line == '':
                break
            line = line[:-1]
            user_data = line.split('|')
            user = {'userId': int(user_data[0]),
                    'age': user_data[1],
                    'gender': user_data[2],
                    'occupation': user_data[3],
                    'zip': user_data[4]
                    }
            users.append(user)
            line = f.readline()

    script = ''
    for i in range(len(users)):
        user = users[i]
        script += f'MERGE (:User_ml100k {{userId:{user["userId"]}, ' \
                  f'                        age:{user["age"]},' \
                  f'                        gender:"{user["gender"]}",' \
                  f'                        occupation:"{user["occupation"]}",' \
                  f'                        zip:"{user["zip"]}"' \
                  f'}})\n'
        if i % 500 == 0:
            tx.run(script)
            script = ''
    tx.run(script)
    print('done user')


def _add_movies(tx):
    with open('../../datasets/movielens/ml-100k/u.item', 'r', encoding="ISO-8859-1") as f:
        line = f.readline()
        while line is not None:
            if line == '':
                break
            line = line[:-1]
            movie_data = line.split('|')
            movie = {
                'movieId': int(movie_data[0]),
                'title': movie_data[1],
                'releaseDate': movie_data[2],
                'videoReleaseDate': movie_data[3],
                'imdbUrl': movie_data[4],
                'genre': [int(movie_data[5]),
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
                          ]
            }
            movies.append(movie)
            line = f.readline()
            '''
            movie id 0| movie title 1| release date 2| video release date 3|
              IMDb URL 4| unknown 5| Action 6| Adventure7 | Animation 8|
              Children's 9| Comedy 10| Crime 11| Documentary 12| Drama 13| Fantasy 14|
              Film-Noir 15| Horror 16| Musical 17| Mystery 18| Romance 19| Sci-Fi 20|
              Thriller 21| War 22| Western 23|
            '''

    script = ''
    for i in range(len(movies)):
        movie = movies[i]
        script += f'MERGE (:Movie_ml100k {{movieId:{movie["movieId"]}, ' \
                  f'                        title:"{movie["title"]}",' \
                  f'                        releaseDate:"{movie["releaseDate"]}",' \
                  f'                        imdbUrl:"{movie["imdbUrl"]}",' \
                  f'                        genre:{movie["genre"]}' \
                  f'}})\n'
        if i % 500 == 0:
            tx.run(script)
            script = ''
    tx.run(script)
    print('done movie')


def _add_ratings(tx):
    with open('../../datasets/movielens/ml-100k/u.data.csv', 'r', encoding="ISO-8859-1") as f:
        line = f.readline()
        while line is not None:
            if line == '':
                break
            line = line[:-1]
            rating_data = line.split('\t')
            user_id = int(rating_data[0])
            movie_id = int(rating_data[1])
            rating = int(rating_data[2])
            timestamp = int(rating_data[3])
            rating = {'userId': user_id,
                      'movieId': movie_id,
                      'rating': rating,
                      'timestamp': timestamp}
            ratings.append(rating)
            line = f.readline()
    script = ''
    for i in range(len(ratings)):
        rating = ratings[i]
        script += f'MATCH (u:User_ml100k {{ userId:{rating["userId"]}  }}), (m:Movie_ml100k {{ movieId:{rating["movieId"]} }}) \n' \
                  f'WITH u,m \n' \
                  f'MERGE (u)-[r:Rating_ml100k {{ rating:{rating["rating"]}, timestamp:{rating["timestamp"]} }}]->(m) ' \
                  f'\n'
    # print(script)
    if i % 500 == 0:
        tx.run(script)
        script = ''
    tx.run(script)
    print('done ratings')


def add_yelp_data(tx):
    _add_yelp_users(tx)
    _add_yelp_business(tx)
    _add_yelp_checkin(tx)
    _add_yelp_tip(tx)
    _add_yelp_review(tx)

def _add_yelp_business(tx):
    pass


def _add_yelp_users(tx):
    pass

def _add_yelp_checkin(tx):
    pass


def _add_yelp_tip(tx):
    pass


def _add_yelp_review(tx):
    pass


if __name__ == '__main__':
    with driver.session() as session:
        session.write_transaction(add_movielens_data)
