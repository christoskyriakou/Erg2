#!/usr/bin/env python3
"""
Neural LSH - Search
Αναζήτηση πλησιέστερων γειτόνων με Neural LSH
"""

import argparse
import sys
import time
import numpy as np
import torch
from pathlib import Path

from src.dataset_parser import DatasetParser
from src.index_manager import IndexManager
from src.search_engine import SearchEngine
from src.metrics import MetricsCalculator


def parse_arguments():
    parser = argparse.ArgumentParser(description='Neural LSH - Search')
    
    # Required arguments
    parser.add_argument('-d', '--dataset', required=True, help='Input dataset file')
    parser.add_argument('-q', '--query', required=True, help='Query file')
    parser.add_argument('-i', '--index', required=True, help='Index path')
    parser.add_argument('-o', '--output', required=True, help='Output file')
    parser.add_argument('-type', '--type', required=True, choices=['sift', 'mnist'],
                        help='Dataset type')
    
    # Optional arguments
    parser.add_argument('-N', '--neighbors', type=int, default=1,
                        help='Number of nearest neighbors (default: 1)')
    parser.add_argument('-R', '--radius', type=float, default=None,
                        help='Range search radius (default: 2000 for MNIST, 2800 for SIFT)')
    parser.add_argument('-T', '--probes', type=int, default=5,
                        help='Number of bins to probe (default: 5)')
    parser.add_argument('-range', '--range_search', type=str, default='true',
                        choices=['true', 'false'],
                        help='Enable range search (default: true)')
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    # Set default radius based on dataset type
    if args.radius is None:
        args.radius = 2000.0 if args.type == 'mnist' else 2800.0
    
    args.range_search = (args.range_search.lower() == 'true')
    
    print("=" * 60)
    print("Neural LSH - Search")
    print("=" * 60)
    
    # Load dataset
    print(f"\n[1/4] Loading dataset from {args.dataset}...")
    parser = DatasetParser()
    data = parser.parse(args.dataset, args.type)
    print(f"  Loaded {len(data)} vectors")
    
    # Load queries
    print(f"\n[2/4] Loading queries from {args.query}...")
    queries = parser.parse(args.query, args.type)
    print(f"  Loaded {len(queries)} queries")
    
    # Load index
    print(f"\n[3/4] Loading index from {args.index}...")
    index_manager = IndexManager()
    index_manager.load(args.index)
    print(f"  Index loaded: {index_manager.n_parts} partitions")
    
    # Initialize search engine
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    search_engine = SearchEngine(index_manager, data, device)
    
    # Perform searches
    print(f"\n[4/4] Performing searches...")
    print(f"  N={args.neighbors}, T={args.probes}, R={args.radius}")
    print(f"  Range search: {args.range_search}")
    
    results = []
    approx_times = []
    true_times = []
    
    for i, query in enumerate(queries):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(queries)}")
        
        # Approximate search
        start_time = time.time()
        neighbors, distances = search_engine.search(
            query, N=args.neighbors, T=args.probes
        )
        approx_time = time.time() - start_time
        approx_times.append(approx_time)
        
        # Range search
        range_neighbors = []
        if args.range_search:
            range_neighbors = search_engine.range_search(
                query, radius=args.radius, T=args.probes
            )
        
        # True search (exhaustive)
        start_time = time.time()
        true_neighbors, true_distances = search_engine.true_search(
            query, N=args.neighbors
        )
        true_time = time.time() - start_time
        true_times.append(true_time)
        
        results.append({
            'query_id': i,
            'neighbors': neighbors,
            'distances': distances,
            'true_neighbors': true_neighbors,
            'true_distances': true_distances,
            'range_neighbors': range_neighbors,
            'approx_time': approx_time,
            'true_time': true_time
        })
    
    # Calculate metrics
    print(f"\n  Computing metrics...")
    metrics_calc = MetricsCalculator()
    avg_af = metrics_calc.calculate_af(results)
    recall = metrics_calc.calculate_recall(results, args.neighbors)
    qps = 1.0 / np.mean(approx_times) if approx_times else 0
    
    # Write results
    print(f"\n  Writing results to {args.output}...")
    write_results(args.output, results, avg_af, recall, qps,
                  np.mean(approx_times), np.mean(true_times))
    
    print(f"\n  Results:")
    print(f"    Average AF: {avg_af:.4f}")
    print(f"    Recall@{args.neighbors}: {recall:.4f}")
    print(f"    QPS: {qps:.2f}")
    print(f"    Avg approx time: {np.mean(approx_times)*1000:.2f}ms")
    print(f"    Avg true time: {np.mean(true_times)*1000:.2f}ms")
    print("=" * 60)


def write_results(output_file, results, avg_af, recall, qps, 
                  t_approx_avg, t_true_avg):
    """Write results to output file"""
    with open(output_file, 'w') as f:
        f.write("Neural LSH\n")
        
        for result in results:
            f.write(f"Query: {result['query_id']}\n")
            
            # Nearest neighbors
            for i, (neighbor, dist, true_dist) in enumerate(zip(
                result['neighbors'], 
                result['distances'],
                result['true_distances']
            )):
                f.write(f"Nearest neighbor-{i+1}: {neighbor}\n")
                f.write(f"distanceApproximate: {dist:.6f}\n")
                f.write(f"distanceTrue: {true_dist:.6f}\n")
            
            # Range neighbors
            f.write("R-near neighbors:\n")
            for neighbor in result['range_neighbors']:
                f.write(f"{neighbor}\n")
        
        # Global metrics
        f.write(f"Average AF: {avg_af:.6f}\n")
        f.write(f"Recall@N: {recall:.6f}\n")
        f.write(f"QPS: {qps:.2f}\n")
        f.write(f"tApproximateAverage: {t_approx_avg:.6f}\n")
        f.write(f"tTrueAverage: {t_true_avg:.6f}\n")


if __name__ == '__main__':
    main()