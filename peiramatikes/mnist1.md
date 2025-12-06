# Πειραματική Μελέτη - Neural LSH

**Generated:** 2025-12-04 18:30:21

kahip mode 2

EXPERIMENTS = {
    'baseline': {
        'knn': 10,
        'm': 100,
        'epochs': 10,
        'layers': 3,
        'nodes': 64,
        'batch_size': 128,
        'lr': 0.001,
        'imbalance': 0.03,
        'T': 5
    },
    'more_bins': {
        'knn': 10,
        'm': 50,
        'epochs': 10,
        'layers': 3,
        'nodes': 64,
        'batch_size': 128,
        'lr': 0.001,
        'imbalance': 0.03,
        'T': 10
    },
    'deeper_network': {
        'knn': 10,
        'm': 50,
        'epochs': 20,
        'layers': 5,
        'nodes': 128,
        'batch_size': 256,
        'lr': 0.001,
        'imbalance': 0.03,
        'T': 10
    },
    'optimal': {
        'knn': 20,
        'm': 30,
        'epochs': 30,
        'layers': 4,
        'nodes': 128,
        'batch_size': 256,
        'lr': 0.0005,
        'imbalance': 0.05,
        'T': 10
    },
    'high_recall': {
        'knn': 25,
        'm': 25,
        'epochs': 40,
        'layers': 4,
        'nodes': 128,
        'batch_size': 256,
        'lr': 0.0005,
        'imbalance': 0.05,
        'T': 12
    }
}


## Summary - All Datasets

| Dataset | Experiment | Recall@10 | Avg AF | QPS | Build Time (min) |
|---------|------------|-----------|--------|-----|------------------|
| mnist | baseline | 95.38% | 1.0029 | 140.30 | 14.01 |
| mnist | optimal | 99.61% | 1.0001 | 21.29 | 12.87 |
| mnist | high_recall | 99.82% | 1.0001 | 15.24 | 13.53 |
| mnist | fast_search | 96.31% | 1.0022 | 104.99 | 13.50 |

---

## MNIST

**Best Configuration:** high_recall (Recall@10: 99.82%)

### Results by Experiment

| Experiment | k | m | epochs | T | Recall@10 | AF | QPS |
|------------|---|---|--------|---|-----------|----|----- |
| baseline | 10 | 100 | 10 | 5 | 95.38% | 1.0029 | 140.30 |
| optimal | 20 | 30 | 30 | 10 | 99.61% | 1.0001 | 21.29 |
| high_recall | 25 | 25 | 40 | 12 | 99.82% | 1.0001 | 15.24 |
| fast_search | 10 | 80 | 15 | 5 | 96.31% | 1.0022 | 104.99 |

### Detailed Results (All N values)

#### baseline


| N | Recall@N | Avg AF | QPS | t_approx (ms) |
|---|----------|--------|-----|---------------|
| 1 | 96.41% | 1.0027 | 131.50 | 7.60 |
| 5 | 95.97% | 1.0027 | 127.88 | 7.82 |
| 10 | 95.38% | 1.0029 | 140.30 | 7.13 |

#### fewer_bins

*Λιγότερα bins (m=50) για μεγαλύτερα partitions*

| N | Recall@N | Avg AF | QPS | t_approx (ms) |
|---|----------|--------|-----|---------------|
| 1 | 99.43% | 1.0003 | 36.59 | 27.33 |
| 5 | 99.26% | 1.0003 | 35.73 | 27.99 |
| 10 | 99.06% | 1.0004 | 33.92 | 29.48 |

#### deeper_net

*Βαθύτερο MLP (5 layers, 128 nodes)*

| N | Recall@N | Avg AF | QPS | t_approx (ms) |
|---|----------|--------|-----|---------------|
| 1 | 99.47% | 1.0003 | 33.67 | 29.70 |
| 5 | 99.30% | 1.0003 | 35.02 | 28.56 |
| 10 | 99.14% | 1.0004 | 35.06 | 28.52 |

#### optimal


| N | Recall@N | Avg AF | QPS | t_approx (ms) |
|---|----------|--------|-----|---------------|
| 1 | 99.81% | 1.0001 | 21.75 | 45.98 |
| 5 | 99.67% | 1.0001 | 22.93 | 43.61 |
| 10 | 99.61% | 1.0001 | 21.29 | 46.96 |

#### high_recall


| N | Recall@N | Avg AF | QPS | t_approx (ms) |
|---|----------|--------|-----|---------------|
| 1 | 99.95% | 1.0000 | 15.14 | 66.06 |
| 5 | 99.87% | 1.0000 | 15.00 | 66.64 |
| 10 | 99.82% | 1.0001 | 15.24 | 65.63 |

#### fast_search

*Γρήγορο search, χαμηλότερο recall*

| N | Recall@N | Avg AF | QPS | t_approx (ms) |
|---|----------|--------|-----|---------------|
| 1 | 97.28% | 1.0019 | 118.23 | 8.46 |
| 5 | 96.84% | 1.0020 | 100.53 | 9.95 |
| 10 | 96.31% | 1.0022 | 104.99 | 9.53 |

---

