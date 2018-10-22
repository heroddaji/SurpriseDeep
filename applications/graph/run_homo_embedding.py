import networkx as nx
from embedding.laplacian_eigenmaps import LaplacianEigenmaps
from embedding.locally_linear_embedding import LocallyLinearEmbedding

def run_embedding():
    G = nx.read_gml('../datasets/zachary/karate.gml',label='id')

    models = []
    models.append(LaplacianEigenmaps(dimension=2))
    models.append(LocallyLinearEmbedding(dimension=2))

    for model in models:
        Y, t = model.learn_embedding(G)
        print(f'{model.name} training time:{t}')
        print(Y)


if __name__ == '__main__':
    run_embedding()
