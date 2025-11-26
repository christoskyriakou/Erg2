import numpy as np
import kahip
from distances import L2_distance_batch
from IVFFLAT import IVFFLAT

def build_knn_graph_with_ivfflat(X, k, nlist=100, nprobe=5):
    n, d = X.shape
    
    # 1. φτιάξε το index
    index = IVFFLAT(d, nlist)
    index.train(X)
    index.add(X)

    # 2. κενός γράφος
    graph = [[] for _ in range(n)]

    # 3. για κάθε vector X[i], τρέξε ANN search
    for i in range(n):
        results = index.search(X[i], k, nprobe)
        
        # κράτα μόνο τα indices
        neighbors = [idx for dist, idx in results]
        graph[i] = neighbors
        
        if i % 100 == 0:
            print(f"[KNN-IVFFLAT] processed {i}/{n}")

    return graph

def make_undirected_weighted(graph):
    n=len(graph)
    undir=[{}for _ in range(n)]
    for i in range(n):
        for j in graph[i]:
            w=2 if i in graph[j] else 1
            undir[i][j]=w
            undir[j][i]=w
    return undir

def graph_to_csr(undir):
    n=len(undir)
    xadj=[0]
    adjncy=[]
    adjcwgt=[]
    for i in range(n):
        for j, w in undir[i].items():
            adjncy.append(j)
            adjcwgt.append(w)
        xadj.append(len(adjncy))
    return xadj, adjncy, adjcwgt

def run_kahip(ugraph, m, imbalance, mode):
    xadj,adjncy,adjcwgt=graph_to_csr(ugraph)
    n=len(ugraph)
    print(f"[KaHIP] Graph: {n} vertices, {len(adjncy)} edges")
    print(f"[KaHIP] Partitioning into {m} parts (imbalance={imbalance})...")
    vwgt=[1]*n
    edgecut,parts=kahip.kaffpa(
        vwgt,
        xadj,
        adjcwgt,
        adjncy,
        int(m),
        float(imbalance),
        True,       # suppress KaHIP internal printing
        0,          # seed
        int(mode)   # mode: KaFFPa, Eco, Fast, Strong, etc.
    )
    print(f"[KaHIP] Edgecut: {edgecut}")
    return np.array(parts, dtype=np.int32)