import networkx as nx
import configuration as conf
import utils as ut
from itertools import combinations


if __name__ == "__main__":
    dataset = ut.load_obj(conf.data_dir, conf.dataset)

    G = nx.Graph()
    for v in dataset:
        if len(v) > 1:
            G.add_edges_from([t for t in combinations(v, 2)])
    ut.save_obj(G, conf.data_dir, f'{conf.dataset}_graph')

    trussness_g = {}
    k = 1

    while k < 100:
        ktruss = nx.k_truss(G, k)
        if len(ktruss) > 0:
            trussness_g[k] = ktruss
            k += 1
        else:
            break
    ut.save_obj(trussness_g, conf.data_dir, f'{conf.dataset}_graph_trussness')
