from neo4j.v1 import GraphDatabase

uri = "bolt://128.199.71.16:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", "DaiNeo4j123!@#"))
_session = None

def get_session():
    global _session
    if _session is None:
        _session = driver.session()
    return _session

def print_friends_of(tx, name):
    for record in tx.run("MATCH (a:Person)-[:KNOWS]->(f) "
                         "WHERE a.name = {name} "
                         "RETURN f.name", name=name):
        print(record["f.name"])


def insert_data(tx, type, json_attrs):
    query = """ MERGE (n:%s %s) 
                ON CREATE SET n.created = timestamp() 
                ON MATCH SET 
                    n.counter = coalesce(n.counter, 0) + 1,
                    n.accessTime = timestamp()""" % (type, json_attrs)
    tx.run(query)
