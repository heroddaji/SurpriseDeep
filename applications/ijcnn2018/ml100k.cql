
CREATE CONSTRAINT ON (u:User_ml100k) ASSERT u.userId IS UNIQUE;
CREATE CONSTRAINT ON (m:Movie_ml100k) ASSERT m.movieId IS UNIQUE;

//read rating data
USING PERIODIC COMMIT
LOAD CSV FROM 'file:///ml100k_old/u.data.csv' AS line
WITH line
MATCH (u:User_ml100k {userId:TOINTEGER(line[0])})
MATCH (m:Movie_ml100k {movieId:TOINTEGER(line[1])})
MERGE (u)-[r:Rating_ml100k {rating:line[2], timestamp:line[3]}]-(m);

