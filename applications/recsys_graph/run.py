import networkx as nx
import matplotlib.pyplot as plt
from graph_datasets import GraphDataset
from embedding.embedding import LaplacianEigenmaps

def get_datasets():
    zachary_ds = GraphDataset('zachary')
    zachary_ds.download()

def process_graph():
    G = nx.read_gml('karate.gml',label='id')

    #embedding
    lap_em = LaplacianEigenmaps(dimension=3)
    lap_Y, lap_time = lap_em.learn(G)
    print(lap_Y)
    print('time:',lap_time)

    #draw embedding
    plt.scatter(lap_Y[:,0],lap_Y[:,1])
    plt.show()

    #show metrics



if __name__ == '__main__':
    # get_datasets()
    process_graph()