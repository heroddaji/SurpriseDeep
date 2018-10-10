import networkx as nx
import scipy.sparse.linalg as lg
import numpy as np
from time import time

class Embedding(object):
    def __init__(self):
        pass


class LaplacianEigenmaps(Embedding):
    def __init__(self, dimension=2):
        super().__init__()
        self.dimension = dimension

    def learn(self, graph):
        t1 = time()

        lap_matrix = nx.normalized_laplacian_matrix(graph)
        w, v = lg.eigs(lap_matrix, k=self.dimension + 1, which='SM')
        t2 = time()
        self._X = v[:, 1:]

        p_d_p_t = np.dot(v, np.dot(np.diag(w), v.T))
        eig_err = np.linalg.norm(p_d_p_t - lap_matrix)
        print('Laplacian matrix recon. error (low rank): %f' % eig_err)
        return self._X, (t2 - t1)