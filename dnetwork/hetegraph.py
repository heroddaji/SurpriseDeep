from .graph import Graph


# undirected heterogeneous graph
class HeteroGraph(Graph):
    _nodes = {}
    _edges = {}

    def add_node(self, node_id: int, type: str, **attr):
        if self._nodes.get(type, None) is None:
            self._nodes[type] = {}
        self._nodes[type][node_id] = attr
        self._nodes[type][node_id]['node_id'] = node_id
        self._nodes[type][node_id]['node_type'] = type
        self._nodes[type][node_id]['node_edges'] = []

    def add_edge(self, node_id_1: int, type1: str, node_id_2: int, type2: str, **attr):
        if self._nodes[type1].get(node_id_1, None) is None:
            raise Exception(f'Cannot find node id {node_id_1} of type {type1}')
        if self._nodes[type2].get(node_id_2, None) is None:
            raise Exception(f'Cannot find node id {node_id_2} of type {type2}')

        edge_type = f'{type1}_{type2}'
        edge_id = f'{node_id_1}_{node_id_2}'

        # use type and id concatenation as key for fast look up
        if self._edges.get(edge_type, None) is None:
            self._edges[edge_type] = {}

        if self._edges[edge_type].get(node_id_1, None) is None:
            self._edges[edge_type][node_id_1] = {}
        if self._edges[edge_type][node_id_1].get(node_id_2, None) is None:
            self._edges[edge_type][node_id_1][node_id_2] = {}
        self._edges[edge_type][node_id_1][node_id_2] = attr

        # #reverse edge
        # edge_type_re = f'{type2}_{type1}'
        # edge_id_re = f'{node_id_2}_{node_id_1}'
        # if self._edges.get(edge_type_re, None) is None:
        #     self._edges[edge_type_re] = {}
        # self._edges[edge_type_re][edge_id_re] = attr
        # self._edges[edge_type_re][edge_id_re]['edge_id'] = edge_id_re
        # self._edges[edge_type_re][edge_id_re]['edge_type'] = edge_type_re
        # self._edges[edge_type_re][edge_id_re]['node_1'] = self._nodes[type1][node_id_1]
        # self._edges[edge_type_re][edge_id_re]['node_2'] = self._nodes[type2][node_id_2]
        # # add edge back to node
        # self._nodes[type1][node_id_1]['node_edges'].append(self._edges[edge_type][edge_id])
        # self._nodes[type2][node_id_2]['node_edges'].append(self._edges[edge_type][edge_id])

    # def get_edge(self, node_id_1: int, type1: str, node_id_2: int, type2: str):
    #     edge_type = f'{type1}_{type2}'
    #     edge_id = f'{node_id_1}_{node_id_2}'
    #     return self._edges[edge_type][edge_id]

    # def get_edges(self, node_id, type):

    @property
    def nodes(self):
        return self._nodes

    @property
    def edges(self):
        return self._edges


if __name__ == '__main__':
    g = HeteroGraph()
    g.add_node(1, type='movie', name='dai1')
    g.add_node(2, type='movie', name='dai2')

    g.add_node(1, type='user', name='nguyen1')
    g.add_node(2, type='user', name='nguyen2')

    g.add_edge(1, 'user', 1, 'movie', weight=4)
    g.add_edge(1, 'user', 2, 'movie', weight=3, timestamp=34)
    from pprint import pprint

    pprint(g.nodes)
    pprint(g.edges['user_movie'][1][2])
