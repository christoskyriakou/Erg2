# Πειραματική Μελέτη - Neural LSH

**Generated:** 2025-12-05 18:12:13

        'knn': 40,            # ← Πολύ πλούσιος γράφος
        'm': 15,              # ← Πολύ λίγα, τεράστια bins
        'epochs': 70,
        'layers': 6,
        'nodes': 320,         # ← Περισσότερη capacity
        'batch_size': 256,
        'lr': 0.00015,        # ← Πολύ προσεκτικό
        'imbalance': 0.12,    # ← Πολύ χαλαρό
        'T': 10,              # ← 67% των bins
        'description': 'Target 70-75% - Very high quality'


## Summary - All Datasets

| Dataset | Experiment | Recall@10 | Avg AF | QPS | Build Time (min) |
|---------|------------|-----------|--------|-----|------------------|
| sift | high_recall | 74.32% | 1.0229 | 29.22 | 9.88 |

---

## SIFT

**Best Configuration:** high_recall (Recall@10: 74.32%)

### Results by Experiment

| Experiment | k | m | epochs | T | Recall@10 | AF | QPS |
|------------|---|---|--------|---|-----------|----|----- |
| high_recall | 40 | 15 | 70 | 10 | 74.32% | 1.0229 | 29.22 |

### Detailed Results (All N values)

#### high_recall

*Target 70-75% - Very high quality*

| N | Recall@N | Avg AF | QPS | t_approx (ms) |
|---|----------|--------|-----|---------------|
| 1 | 74.78% | 1.0000 | 29.25 | 34.19 |
| 10 | 74.32% | 1.0229 | 29.22 | 34.22 |

---

