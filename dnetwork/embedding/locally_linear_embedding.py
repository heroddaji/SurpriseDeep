from .embedding_base import EmbeddingBase
import scipy.sparse as sp
import scipy.sparse.linalg as lg
from sklearn.preprocessing import normalize
from time import time
import networkx as nx
import numpy as np


class LocallyLinearEmbedding(EmbeddingBase):
    name = 'LocallyLinearEmbedding'

    def __init__(self, **kwargs):
        super().__init__()
        self.dimension = kwargs.get('dimension', None)

        if self.dimension is None:
            raise Exception("Need dimension param")

    def learn_embedding(self, graph):
        graph = graph.to_undirected()
        t1 = time()
        A = nx.to_scipy_sparse_matrix(graph)
        normalize(A, norm='l1', axis=1, copy=False)
        I_n = sp.eye(graph.number_of_nodes())
        I_min_A = I_n - A
        u, s, vt = lg.svds(I_min_A, k=self.dimension + 1, which='SM')
        t2 = time()
        self._X = vt.T
        self._X = self._X[:, 1:]
        return self._X, (t2 - t1)

    def get_embedding(self):
        return self._X

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
