import argparse
import numpy as np
from dataset_parser import load_dataset
from distances import L2_distance_batch
from graph_utils import (
    build_knn_graph,
    make_undirected_weighted,
    run_kahip
)
import os


# -------------------------------------------------------
# MAIN BUILD FUNCTION
# -------------------------------------------------------
def main():

    parser = argparse.ArgumentParser(description="Neural LSH Build")

    parser.add_argument("-d", "--data", required=True, help="Input dataset file")
    parser.add_argument("-i", "--index", required=True, help="Output index directory")
    parser.add_argument("-type", "--type", required=True, choices=["sift", "mnist"],
                        help="Dataset type")

    parser.add_argument("--knn", type=int, default=10, help="k for k-NN graph")
    parser.add_argument("-m", type=int, default=100, help="Number of blocks for KaHIP")
    parser.add_argument("--imbalance", type=float, default=0.03, help="KaHIP imbalance")
    parser.add_argument("--kahip_mode", type=int, default=2, help="KaHIP mode: 0,1,2")

    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--nodes", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=1)

    args = parser.parse_args()

    # ---------------------------------------------------
    # 1. Load dataset
    # ---------------------------------------------------
    print("[BUILD] Loading dataset...")
    X, dtype = load_dataset(args.data)

    print(f"[BUILD] Loaded dataset: {X.shape[0]} vectors, dim={X.shape[1]}")
    print(f"[BUILD] Type: {dtype}")

    # ---------------------------------------------------
    # 2. Build kNN graph
    # ---------------------------------------------------
    print(f"[BUILD] Building KNN graph (k={args.knn})...")
    graph = build_knn_graph(X, k=args.knn)

    # ---------------------------------------------------
    # 3. Make undirected weighted graph
    # ---------------------------------------------------
    print("[BUILD] Making undirected + weighted graph...")
    ugraph = make_undirected_weighted(graph)

    # ---------------------------------------------------
    # 4. Run KaHIP
    # ---------------------------------------------------
    print(f"[BUILD] Running KaHIP (m={args.m}, imbalance={args.imbalance})...")
    parts = run_kahip(ugraph, args.m, args.imbalance, args.kahip_mode)

    print("[BUILD] KaHIP finished.")
    print(f"[BUILD] Example partitions: {parts[:20]}")

    # ---------------------------------------------------
    # 5. Save graph partitions for MLP training
    # ---------------------------------------------------
    os.makedirs(args.index, exist_ok=True)
    out_path = os.path.join(args.index, "partitions.npy")

    np.save(out_path, parts)

    print(f"[BUILD] Saved partitions to {out_path}")

    print("[BUILD] Done.")


# -------------------------------------------------------
# ENTRY POINT
# -------------------------------------------------------
if __name__ == "__main__":
    main()
