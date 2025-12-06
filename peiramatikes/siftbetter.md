# Πειραματική Μελέτη - Neural LSH

**Generated:** 2025-12-05 15:05:39

## Summary - All Datasets

    'baseline': {
        'knn': 10,
        'm': 100,
        'epochs': 10,
        'layers': 3,
        'nodes': 64,
        'batch_size': 128,
        'lr': 0.001,
        'imbalance': 0.03,
        'T': 5,
        'description': 'Quick baseline με default params'
    },
    'fewer_bins': {
        'knn': 15,
        'm': 40,
        'epochs': 20,
        'layers': 4,
        'nodes': 128,
        'batch_size': 256,
        'lr': 0.001,
        'imbalance': 0.04,
        'T': 12,
        'description': 'Λιγότερα bins (m=40) για μεγαλύτερα partitions'
    },
    'deeper_net': {
        'knn': 20,
        'm': 35,
        'epochs': 25,
        'layers': 5,
        'nodes': 128,
        'batch_size': 256,
        'lr': 0.0008,
        'imbalance': 0.05,
        'T': 14,
        'description': 'Βαθύτερο MLP (5 layers, 128 nodes)'
    },
    'optimal': {
        'knn': 25,
        'm': 30,
        'epochs': 35,
        'layers': 4,
        'nodes': 128,
        'batch_size': 256,
        'lr': 0.0005,
        'imbalance': 0.05,
        'T': 12,
        'description': 'Βέλτιστο balance recall/QPS'
    },
    'high_recall': {
        'knn': 30,
        'm': 25,
        'epochs': 50,
        'layers': 5,
        'nodes': 256,
        'batch_size': 256,
        'lr': 0.0003,
        'imbalance': 0.05,
        'T': 15,
        'description': 'Μέγιστο recall (αργότερο)'
    },
    'fast_search': {
        'knn': 10,
        'm': 80,
        'epochs': 15,
        'layers': 3,
        'nodes': 64,
        'batch_size': 256,
        'lr': 0.001,
        'imbalance': 0.03,
        'T': 6,
        'description': 'Γρήγορο search, χαμηλότερο recall'
    }
}


| Dataset | Experiment | Recall@10 | Avg AF | QPS | Build Time (min) |
|---------|------------|-----------|--------|-----|------------------|
| sift | baseline | 19.62% | 1.1217 | 101.02 | 6.77 |
| sift | fewer_bins | 30.56% | 1.0834 | 71.28 | 7.10 |
| sift | deeper_net | 43.37% | 1.0641 | 58.07 | 7.11 |
| sift | optimal | 39.69% | 1.0598 | 53.93 | 7.39 |
| sift | high_recall | 61.62% | 1.0332 | 37.31 | 9.27 |
| sift | fast_search | 7.58% | 1.1777 | 177.25 | 7.22 |

---

## SIFT

**Best Configuration:** high_recall (Recall@10: 61.62%)

### Results by Experiment

| Experiment | k | m | epochs | T | Recall@10 | AF | QPS |
|------------|---|---|--------|---|-----------|----|----- |
| baseline | 15 | 100 | 20 | 20 | 19.62% | 1.1217 | 101.02 |
| fewer_bins | 15 | 40 | 20 | 12 | 30.56% | 1.0834 | 71.28 |
| deeper_net | 20 | 35 | 25 | 14 | 43.37% | 1.0641 | 58.07 |
| optimal | 25 | 30 | 35 | 12 | 39.69% | 1.0598 | 53.93 |
| high_recall | 30 | 25 | 50 | 15 | 61.62% | 1.0332 | 37.31 |
| fast_search | 10 | 80 | 15 | 6 | 7.58% | 1.1777 | 177.25 |

### Detailed Results (All N values)

#### baseline

*Γρήγορο search, χαμηλότερο recall*

| N | Recall@N | Avg AF | QPS | t_approx (ms) |
|---|----------|--------|-----|---------------|
| 1 | 20.49% | 1.0002 | 111.16 | 9.00 |
| 5 | 19.79% | 1.1516 | 111.36 | 8.98 |
| 10 | 19.62% | 1.1217 | 101.02 | 9.90 |

#### fewer_bins

*Γρήγορο search, χαμηλότερο recall*

| N | Recall@N | Avg AF | QPS | t_approx (ms) |
|---|----------|--------|-----|---------------|
| 1 | 30.70% | 1.0001 | 71.81 | 13.93 |
| 5 | 30.72% | 1.1035 | 74.74 | 13.38 |
| 10 | 30.56% | 1.0834 | 71.28 | 14.03 |

#### deeper_net

*Γρήγορο search, χαμηλότερο recall*

| N | Recall@N | Avg AF | QPS | t_approx (ms) |
|---|----------|--------|-----|---------------|
| 1 | 41.61% | 1.0001 | 56.80 | 17.60 |
| 5 | 43.12% | 1.0840 | 57.37 | 17.43 |
| 10 | 43.37% | 1.0641 | 58.07 | 17.22 |

#### optimal

*Γρήγορο search, χαμηλότερο recall*

| N | Recall@N | Avg AF | QPS | t_approx (ms) |
|---|----------|--------|-----|---------------|
| 1 | 42.71% | 1.0001 | 57.58 | 17.37 |
| 5 | 40.25% | 1.0697 | 55.25 | 18.10 |
| 10 | 39.69% | 1.0598 | 53.93 | 18.54 |

#### high_recall

*Γρήγορο search, χαμηλότερο recall*

| N | Recall@N | Avg AF | QPS | t_approx (ms) |
|---|----------|--------|-----|---------------|
| 1 | 63.02% | 1.0001 | 36.38 | 27.49 |
| 5 | 61.96% | 1.0397 | 35.37 | 28.27 |
| 10 | 61.62% | 1.0332 | 37.31 | 26.80 |

#### fast_search

*Γρήγορο search, χαμηλότερο recall*

| N | Recall@N | Avg AF | QPS | t_approx (ms) |
|---|----------|--------|-----|---------------|
| 1 | 7.39% | 1.0003 | 187.06 | 5.35 |
| 5 | 7.58% | 1.2145 | 176.83 | 5.66 |
| 10 | 7.58% | 1.1777 | 177.25 | 5.64 |

---

