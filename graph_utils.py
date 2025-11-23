import numpy as np
import kahip
from distances import L2_distance_batch


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
            w = 2 if i in graph[j] else 1
            undirected[i][j] = w
            undirected[j][i] = w

    return undirected


# --------------------------------------------------------
# 3) Convert to CSR format
# --------------------------------------------------------
def graph_to_csr(ugraph):
    """Convert undirected weighted graph to CSR format"""
    n = len(ugraph)
    xadj = [0]
    adjncy = []
    adjcwgt = []

    for i in range(n):
        for j, w in ugraph[i].items():
            adjncy.append(j)
            adjcwgt.append(w)
        xadj.append(len(adjncy))

    return xadj, adjncy, adjcwgt


# --------------------------------------------------------
# 4) Run KaHIP partitioning
# --------------------------------------------------------
def run_kahip(ugraph, m, imbalance, mode):
    xadj, adjncy, adjcwgt = graph_to_csr(ugraph)
    n = len(ugraph)

    print(f"[KaHIP] Graph: {n} vertices, {len(adjncy)} edges")
    print(f"[KaHIP] Partitioning into {m} parts (imbalance={imbalance})...")
    
    # KaHIP expects: vwgt, xadj, adjcwgt, adjncy, nparts, imbalance, suppress, seed, mode
    # Based on pybind11 signature with 9 arguments
    vwgt = [1] * n  # vertex weights
    
    # Try multiple argument combinations
    try:
        # Attempt 1: Standard order with edge weights
        result = kahip.kaffpa(
            vwgt,
            xadj,
            adjcwgt,
            adjncy,
            m,
            float(imbalance),
            True,
            0,
            int(mode)
        )
        print(f"[KaHIP] Partitioning finished! Result type: {type(result)}")
        
        # KaHIP returns (edgecut, partition_array)
        if isinstance(result, tuple):
            edgecut, parts = result
            print(f"[KaHIP] Edge cut: {edgecut}")
            return np.array(parts, dtype=np.int32)
        else:
            return np.array(result, dtype=np.int32)
    except Exception as e:
        print(f"[ERROR] Attempt 1 failed: {e}")
        # If it worked but failed on conversion, the other attempts won't help
        raise RuntimeError(f"KaHIP succeeded but result format unexpected: {e}")