import argparse
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from models import MLP
from dataset_parser import load_dataset
from graph_utils import (
    build_knn_graph,
    make_undirected_weighted,
    run_kahip
)

import os


# -------------------------------------------------------
# Train MLP classifier that predicts KaHIP partitions
# -------------------------------------------------------
def train_mlp(X, parts, args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MLP] Using device: {device}")

    n, d = X.shape
    m = args.m

    # Convert to torch tensors
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(parts).long()

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Create MLP model
    model = MLP(
        input_dim=d,
        num_classes=m,
        layers=args.layers,
        hidden=args.nodes
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("[MLP] Training...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)

        avg_loss = total_loss / n
        print(f"[MLP] Epoch {epoch+1}/{args.epochs} - loss={avg_loss:.4f}")

    print("[MLP] Training finished.")
    return model


# -------------------------------------------------------
# MAIN BUILD PIPELINE
# -------------------------------------------------------
def main():

    parser = argparse.ArgumentParser(description="Neural LSH Build")

    parser.add_argument("-d", "--data", required=True, help="Input dataset file")
    parser.add_argument("-i", "--index", required=True, help="Output index directory")
    parser.add_argument("-type", "--type", required=True, choices=["sift", "mnist"],
                        help="Dataset type")

    parser.add_argument("--knn", type=int, default=10)
    parser.add_argument("-m", type=int, default=100)
    parser.add_argument("--imbalance", type=float, default=0.03)
    parser.add_argument("--kahip_mode", type=int, default=2)

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
    print("[BUILD] Building undirected + weighted graph...")
    ugraph = make_undirected_weighted(graph)

    # ---------------------------------------------------
    # 4. Run KaHIP
    # ---------------------------------------------------
    print(f"[BUILD] Running KaHIP (m={args.m}, imbalance={args.imbalance})...")
    parts = run_kahip(ugraph, args.m, args.imbalance, args.kahip_mode)

    print("[BUILD] KaHIP finished.")
    print(f"[BUILD] Sample partitions: {parts[:20]}")

    # Ensure index directory exists
    os.makedirs(args.index, exist_ok=True)

    # Save partitions
    parts_path = os.path.join(args.index, "partitions.npy")
    np.save(parts_path, parts)
    print(f"[BUILD] Saved partitions → {parts_path}")

    # ---------------------------------------------------
    # 5. Train MLP classifier
    # ---------------------------------------------------
    print("[BUILD] Starting MLP training...")
    model = train_mlp(X, parts, args)

    # ---------------------------------------------------
    # 6. Build inverted file: block_id → list of point indices
    # ---------------------------------------------------
    print("[BUILD] Building inverted index...")
    inverted = {r: [] for r in range(args.m)}
    for idx, r in enumerate(parts):
        inverted[int(r)].append(int(idx))

    # ---------------------------------------------------
    # 7. SAVE INDEX FILES
    # ---------------------------------------------------

    # Save model
    model_path = os.path.join(args.index, "model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"[BUILD] Saved model → {model_path}")

    # Save inverted index
    inv_path = os.path.join(args.index, "inverted_index.npy")
    np.save(inv_path, np.array(inverted, dtype=object))
    print(f"[BUILD] Saved inverted index → {inv_path}")

    print("[BUILD] All tasks completed successfully.")


# -------------------------------------------------------
# ENTRY POINT
# -------------------------------------------------------
if __name__ == "__main__":
    main()