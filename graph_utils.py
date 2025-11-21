from distances import L2_distance_batch
import numpy as np
import kahip


# --------------------------------------------------------
# 1) Directed kNN graph
# --------------------------------------------------------
def build_knn_graph(X, k=10): 
    n = X.shape[0] 
    graph = [[] for _ in range(n)]

    for i in range(n):
        dists = L2_distance_batch(X, X[i])
        idx = np.argpartition(dists, k+1)[:k+1]
        idx = idx[idx != i]
        graph[i] = idx[:k].tolist()

        if i % 100 == 0:
            print(f"[KNN] Processed {i}/{n}")

    return graph


# --------------------------------------------------------
# 2) Undirected weighted graph
# --------------------------------------------------------
def make_undirected_weighted(graph):
    n = len(graph)
    undirected = [{} for _ in range(n)]

    for i in range(n):
        for j in graph[i]:
            if i in graph[j]:
                w = 2      # mutual
            else:
                w = 1      # non mutual

            undirected[i][j] = w
            undirected[j][i] = w

    return undirected


# --------------------------------------------------------
# 3) Convert to CSR
# --------------------------------------------------------
def graph_to_csr(graph):
    n = len(graph)
    xadj = [0]
    adjncy = []
    adjcwgt = []
    vwgt = [1] * n

    for i in range(n):
        for j, w in graph[i].items():
            adjncy.append(j)
            adjcwgt.append(w)
        xadj.append(len(adjncy))

    return vwgt, xadj, adjncy, adjcwgt


# --------------------------------------------------------
# 4) Run KaHIP
# --------------------------------------------------------
def run_kahip(ugraph, m, imbalance, mode):
    vwgt, xadj, adjncy, adjcwgt = graph_to_csr(ugraph)
    n = len(ugraph)

    print("[KaHIP] Running partitioning...")

    parts = kahip.kaffpa(
        n,
        xadj,
        adjncy,
        vwgt,
        adjcwgt,
        nparts=m,
        imbalance=imbalance,
        mode=mode
    )

    return parts
