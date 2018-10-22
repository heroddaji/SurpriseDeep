from time import time
import numpy as np
import networkx as nx
import scipy.sparse.linalg as lg

from .embedding_base import EmbeddingBase


class LaplacianEigenmaps(EmbeddingBase):
    name = 'LaplacianEigenmaps'

    def __init__(self, **kwargs):
        super().__init__()
        self.dimension = kwargs.get('dimension',None)

        if self.dimension is None:
            raise Exception("Need dimension param")

    def learn_embedding(self, graph):
        graph_un = graph.to_undirected()
        t1 = time()
        lap_matrix = nx.normalized_laplacian_matrix(graph_un)

        w, v = lg.eigs(lap_matrix, k = self.dimension + 1, which='SM')
        t2 = time()
        self._X = v[:, 1:]

        p_d_p_t = np.dot(v, np.dot(np.diag(w), v.T))
        eig_err = np.linalg.norm(p_d_p_t - lap_matrix)
        print('Laplacian matrix recon. error (low rank): %f' % eig_err)
        return self._X, (t2 - t1)

    def get_embedding(self):
        return  self._X

    def get_edge_weight(self, i, j):
        return np.exp(
            -np.power(np.linalg.norm(self._X[i, :] - self._X[j, :]), 2)
        )

    def get_reconstructed_adj(self, X=None, node_l=None):
        if X is not None:
            node_num = X.shape[0]
            self._X = X
        else:
            node_num = self._node_num
        adj_mtx_r = np.zeros((node_num, node_num))
        for v_i in range(node_num):
            for v_j in range(node_num):
                if v_i == v_j:
                    continue
                adj_mtx_r[v_i, v_j] = self.get_edge_weight(v_i, v_j)
        return adj_mtx_r
