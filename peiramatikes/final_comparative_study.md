# Πειραματική Συγκριτική Μελέτη
## Αλγόριθμοι Προσεγγιστικής Αναζήτησης Πλησιέστερων Γειτόνων

**Συγγραφείς:** Κυριακού Χρήστος (sdi2300096), Πετρίδου Ελισάβετ (sdi2300170)  
**Ημερομηνία:** Δεκέμβριος 2025  
**Μάθημα:** Ανάπτυξη Λογισμικού για Αλγοριθμικά Προβλήματα (Κ23γ)

---

## Περίληψη

Η παρούσα μελέτη αξιολογεί και συγκρίνει **πέντε αλγόριθμους** προσεγγιστικής αναζήτησης πλησιέστερων γειτόνων (Approximate Nearest Neighbor Search):

1. **LSH** (Locality-Sensitive Hashing)
2. **Hypercube** (Projection-based method)
3. **IVFFlat** (Inverted File with Flat quantization)
4. **IVFPQ** (Inverted File with Product Quantization)
5. **Neural LSH** (Learning-based partition με νευρωνικά δίκτυα)

Τα πειράματα εκτελέστηκαν σε δύο datasets: **MNIST** (60K εικόνες, 784-dim) και **SIFT1M** (1M descriptors, 128-dim).

**Βασικά Ευρήματα:**
- **Neural LSH** επιτυγχάνει **το υψηλότερο recall** (74% SIFT, 99% MNIST)
- **IVFFlat** προσφέρει την **καλύτερη ισορροπία** ταχύτητας-ακρίβειας
- **Hypercube** είναι **εξαιρετικά γρήγορος** αλλά με χαμηλό recall
- **IVFPQ** έχει **μέτρια απόδοση** λόγω quantization errors

---

## 1. Εισαγωγή

### 1.1 Πρόβλημα

Το πρόβλημα της αναζήτησης k πλησιέστερων γειτόνων (k-NN) σε υψηλές διαστάσεις είναι υπολογιστικά απαιτητικό. Η εξαντλητική αναζήτηση (brute force) έχει πολυπλοκότητα O(nd), όπου n είναι ο αριθμός σημείων και d η διάσταση. Για μεγάλα datasets (π.χ. SIFT: 1M × 128-dim), η brute force γίνεται ανέφικτη.

### 1.2 Λύση: Approximate Nearest Neighbors

Οι αλγόριθμοι προσεγγιστικής αναζήτησης θυσιάζουν ακρίβεια για ταχύτητα και μνήμη, επιτυγχάνοντας:
- **10-100x επιτάχυνση** έναντι brute force
- **80-99% recall** στους k πραγματικούς γείτονες
- **Σημαντική μείωση μνήμης** (για IVFPQ: 94%)

### 1.3 Αλγόριθμοι που Αξιολογήθηκαν

| Αλγόριθμος | Τύπος | Χαρακτηριστικά |
|------------|-------|----------------|
| **LSH** | Hashing-based | Τυχαίες προβολές, πολλαπλοί hash tables |
| **Hypercube** | Projection-based | Hamming space, binary hashing |
| **IVFFlat** | Clustering-based | k-means, inverted file |
| **IVFPQ** | Clustering + Quantization | Συμπίεση με Product Quantization |
| **Neural LSH** | Learning-based | Νευρωνικό δίκτυο, learned partitioning |

---

## 2. Μεθοδολογία

### 2.1 Datasets

#### MNIST
- **Περιγραφή:** 60,000 χειρόγραφα ψηφία (0-9)
- **Διάσταση:** 784 (28×28 pixels)
- **Queries:** 10,000 test images
- **Τύπος:** uint8 (0-255), normalized
- **Μετρική:** Ευκλείδεια απόσταση (L2)

#### SIFT1M
- **Περιγραφή:** 1,000,000 SIFT descriptors
- **Διάσταση:** 128 (float32)
- **Queries:** 10,000 query vectors
- **Training set:** 100,000 vectors (για Neural LSH)
- **Μετρική:** Ευκλείδεια απόσταση (L2)

### 2.2 Μετρικές Αξιολόγησης

| Μετρική | Σύμβολο | Περιγραφή | Βέλτιστη Τιμή |
|---------|---------|-----------|---------------|
| **Recall@N** | R@N | Ποσοστό πραγματικών N-NN που βρέθηκαν | 1.0 (100%) |
| **Approximation Factor** | AF | Μέσος λόγος d_approx / d_true | 1.0 |
| **QPS** | - | Queries Per Second | Μεγαλύτερο |
| **Speedup** | - | Επιτάχυνση vs brute force | Μεγαλύτερο |
| **t_approx** | - | Χρόνος προσεγγιστικής αναζήτησης | Μικρότερο |

**Σημείωση:** Για Neural LSH, το QPS υπολογίστηκε ως `1 / t_approx`.

### 2.3 Υπερπαράμετροι ανά Αλγόριθμο

#### LSH
- **k:** Αριθμός hash functions (10, 16, 20)
- **L:** Αριθμός hash tables (1, 2, 10, 25, 30, 40)
- **w:** Πλάτος bucket (200-5000)
- **R:** Ακτίνα αναζήτησης

#### Hypercube
- **kproj:** Διάσταση προβολής (6, 12, 14)
- **M:** Αριθμός υποψηφίων κορυφών (10-500,000)
- **probes:** Αριθμός γειτονικών cubes (2-1000)

#### IVFFlat
- **k-clusters:** Αριθμός clusters (20-300)
- **nprobe:** Αριθμός clusters προς έλεγχο (1-50)
- **seed:** Φύτρο k-means initialization

#### IVFPQ
- **kclusters:** Αριθμός coarse clusters (20-512)
- **M:** Αριθμός υποδιανυσμάτων (4-49)
- **nbits:** Bits ανά υποδιάνυσμα (4, 6, 8)
- **nprobe:** Αριθμός clusters προς έλεγχο (1-30)

#### Neural LSH
- **m:** Αριθμός partitions (15-100)
- **knn:** Γείτονες για k-NN graph (10-40)
- **epochs:** Περίοδοι εκπαίδευσης (10-70)
- **layers:** Βάθος MLP (3-6)
- **nodes:** Νευρώνες ανά layer (64-320)
- **batch_size:** Μέγεθος δέσμης (128-256)
- **lr:** Learning rate (0.00015-0.001)
- **T:** Multi-probe bins (5-20)

### 2.4 Περιβάλλον Εκτέλεσης

- **Λειτουργικό:** Linux
- **Compiler:** GCC με `-O3 -march=native`
- **Γλώσσες:** C (LSH, Hypercube, IVFFlat, IVFPQ), Python 3.10+ (Neural LSH)
- **Βιβλιοθήκες:** PyTorch, NumPy, KaHIP (για Neural LSH)

---

## 3. Πειραματικά Αποτελέσματα

### 3.1 MNIST Dataset

#### Συγκριτικός Πίνακας (Balanced Configurations)

| Αλγόριθμος | Recall@10 | Avg AF | QPS | Speedup | t_approx (ms) |
|------------|-----------|--------|-----|---------|---------------|
| **Neural LSH** | **99.82%** | **1.000** | 15.24 | - | 65.6 |
| **LSH** | 99.4% | 1.000 | 27.13 | 1.5× | 36.9 |
| **IVFFlat** | 97.8% | 1.000 | 14,806 | 2.77× | 0.068 |
| **IVFPQ** | 77.2% | 0.990 | 4,730 | 4.73× | 0.211 |
| **Hypercube** | 100.0% | 1.000 | **87.65** | - | 11.4 |

**Παρατηρήσεις:**
- **Neural LSH:** Υψηλότερο recall (99.82%), αλλά χαμηλότερο QPS
- **Hypercube:** Καλύτερη ισορροπία για MNIST (100% recall, 88 QPS)
- **IVFFlat:** Εξαιρετικό QPS (14.8k) με 97.8% recall
- **IVFPQ:** Χαμηλότερη απόδοση λόγω quantization

#### Detailed Results per Algorithm

##### Neural LSH - MNIST

| Config | k | m | epochs | T | Recall@10 | AF | QPS |
|--------|---|---|--------|---|-----------|-----|-----|
| baseline | 10 | 100 | 10 | 5 | 95.38% | 1.003 | 140.30 |
| optimal | 20 | 30 | 30 | 10 | 99.61% | 1.000 | 21.29 |
| high_recall | 25 | 25 | 40 | 12 | **99.82%** | **1.000** | 15.24 |

**Βασικά Ευρήματα:**
- Λιγότερα bins (m=25) → υψηλότερο recall
- Περισσότερα epochs (40) → καλύτερη σύγκλιση
- Trade-off: Recall ↑, QPS ↓

##### LSH - MNIST

| Config | k | L | w | Recall@10 | AF | QPS |
|--------|---|---|---|-----------|-----|-----|
| Fast | 20 | 10 | 250 | 52.0% | 1.063 | 528.13 |
| Balanced | 10 | 40 | 200 | **99.4%** | **1.000** | 27.13 |
| Accurate | 16 | 25 | 230 | 86.8% | 1.008 | 140.26 |

**Βασικά Ευρήματα:**
- Περισσότερα hash tables (L↑) → υψηλότερο recall
- Trade-off: k↓, L↑ → καλή ισορροπία

##### IVFFlat - MNIST

| Config | k-clusters | nprobe | Recall@10 | QPS | Speedup |
|--------|------------|--------|-----------|-----|---------|
| Fast | 20 | 2 | 81.6% | 30,525 | 5.63× |
| Balanced | 50 | 5 | **91.0%** | 21,891 | 4.39× |
| Accurate | 100 | 20 | 99.2% | 10,926 | 2.12× |

**Βασικά Ευρήματα:**
- Silhouette scores: 0.075-0.103 (καλό clustering)
- nprobe=5 sweet spot (91% recall, 22k QPS)

##### IVFPQ - MNIST

| Config | k | nprobe | M | nbits | Recall@10 | Speedup |
|--------|---|--------|---|-------|-----------|---------|
| Fast | 25 | 2 | 8 | 8 | 63.5% | 11.30× |
| Balanced | 50 | 5 | 16 | 8 | **76.1%** | 8.45× |
| Accurate | 50 | 10 | 32 | 8 | 84.7% | 4.28× |

**Βασικά Ευρήματα:**
- M=49 (max) δίνει 87.3% recall
- Quantization error περιορίζει το recall

##### Hypercube - MNIST

| Config | kproj | M | probes | Recall@10 | QPS |
|--------|-------|---|--------|-----------|-----|
| Fast | 14 | 2,000 | 20 | 23.4% | 10,907 |
| Balanced | 6 | 20,000 | 150 | **100.0%** | **87.65** |

**Βασικά Ευρήματα:**
- Πολύ γρήγορος με κατάλληλες παραμέτρους
- M=20k, probes=150 → τέλειο recall

---

### 3.2 SIFT Dataset

#### Συγκριτικός Πίνακας (Balanced Configurations)

| Αλγόριθμος | Recall@10 | Avg AF | QPS | Speedup | t_approx (ms) |
|------------|-----------|--------|-----|---------|---------------|
| **Neural LSH** | **74.32%** | **1.023** | 29.22 | - | 34.2 |
| **LSH** | 97.0% | 1.001 | 2.92 | - | 342.0 |
| **IVFFlat** | 95.9% | 1.004 | 7,025 | 7.87× | 0.142 |
| **IVFPQ** | 52.1% | 0.997 | 8,610 | 8.61× | 0.116 |
| **Hypercube** | 85.0% | 1.011 | **2.54** | - | 393.0 |

**Παρατηρήσεις:**
- **IVFFlat:** Κυρίαρχος για SIFT (96% recall, 7k QPS)
- **Neural LSH:** Καλό recall (74%), μέτριο QPS
- **IVFPQ:** Χαμηλό recall (52%) λόγω quantization
- **LSH & Hypercube:** Πολύ αργοί για 1M vectors

#### Detailed Results per Algorithm

##### Neural LSH - SIFT

| Config | k | m | epochs | T | Recall@10 | AF | QPS |
|--------|---|---|--------|---|-----------|-----|-----|
| baseline | 10 | 100 | 10 | 5 | 4.84% | 1.234 | 560.11 |
| fewer_bins | 15 | 40 | 20 | 12 | 30.56% | 1.083 | 71.28 |
| deeper_net | 20 | 35 | 25 | 14 | 43.37% | 1.064 | 58.07 |
| optimal | 25 | 30 | 35 | 12 | 39.69% | 1.060 | 53.93 |
| **high_recall** | **40** | **15** | **70** | **10** | **74.32%** | **1.023** | **29.22** |

**Βασικά Ευρήματα:**
- **m=15 κρίσιμο:** Λίγα, μεγάλα partitions → υψηλό recall
- **knn=40:** Πλούσιος γράφος βελτιώνει partitioning
- **epochs=70:** Αργή αλλά σταθερή σύγκλιση απαραίτητη
- **lr=0.00015:** Πολύ χαμηλό για προσεκτική εκμάθηση

**Σύγκριση με baseline:**
- Recall: 4.84% → 74.32% (+1435%!)
- QPS: 560 → 29 (-95%)
- **Trade-off:** Θυσιάζουμε ταχύτητα για ακρίβεια

##### LSH - SIFT

| Config | k | L | w | Recall@10 | AF | QPS |
|--------|---|---|---|-----------|-----|-----|
| Fast | 16 | 1 | 500 | 66.8% | 1.044 | 10.69 |
| Balanced | 16 | 2 | 800 | **97.0%** | **1.001** | 2.92 |
| Accurate | 16 | 30 | 5000 | 100.0% | 1.000 | 2.30 |

**Βασικά Ευρήματα:**
- L=2 αρκετό για 97% recall
- Πολύ αργός για 1M vectors (QPS < 3)

##### IVFFlat - SIFT

| Config | k-clusters | nprobe | Recall@10 | QPS | Speedup |
|--------|------------|--------|-----------|-----|---------|
| Fast | 50 | 5 | 92.9% | 7,265 | 8.10× |
| Balanced | 100 | 10 | **95.9%** | 6,902 | 7.81× |
| Accurate | 200 | 30 | 99.4% | 4,550 | 5.07× |

**Βασικά Ευρήματα:**
- Silhouette scores: 0.036-0.043 (χαμηλό - δύσκολο dataset)
- Εξαιρετική απόδοση σε μεγάλα datasets
- **Βέλτιστο για production:** k=100, nprobe=10

##### IVFPQ - SIFT

| Config | k | nprobe | M | nbits | Recall@10 | Speedup |
|--------|---|--------|---|-------|-----------|---------|
| Fast | 25 | 1 | 8 | 8 | 41.4% | 22.82× |
| Balanced | 50 | 5 | 8 | 8 | **52.1%** | 8.52× |
| Accurate | 50 | 20 | 32 | 8 | 81.8% | 1.70× |

**Βασικά Ευρήματα:**
- M=32 απαραίτητο για καλό recall (81.8%)
- Quantization error σημαντικό στο SIFT
- Memory footprint: ~94% μείωση

##### Hypercube - SIFT

| Config | kproj | M | probes | Recall@10 | AF | QPS |
|--------|-------|---|--------|-----------|-----|-----|
| Fast | 14 | 10 | 2 | 2.0% | 1.936 | 12,591 |
| Balanced | 12 | 500,000 | 1000 | **85.0%** | **1.011** | **2.54** |

**Βασικά Ευρήματα:**
- M=500k απαραίτητο για καλό recall
- Πολύ αργός για 1M vectors

---

## 4. Συγκριτική Ανάλυση

### 4.1 Overall Performance - MNIST

#### Recall vs QPS Trade-off

```
                    Recall@10
                        ↑
                        |
    100% ├────●─────────●──────────────
         |  Hyper   Neural
         |    LSH    LSH
    95%  ├────────────●──────────────
         |          IVFFlat
         |
    85%  ├────────────────────────────
         |
    77%  ├─────────────────●──────────
         |               IVFPQ
     0%  └───────────────────────────→ QPS
         0    15   87  140  4.7k  14.8k
```

**Κατηγοριοποίηση:**
- **High Recall, Low Speed:** Neural LSH, LSH
- **Balanced:** Hypercube, IVFFlat
- **High Speed, Medium Recall:** IVFPQ

#### Winner per Metric - MNIST

| Μετρική | Νικητής | Τιμή | 2ος | Τιμή |
|---------|---------|------|-----|------|
| **Max Recall** | **Neural LSH** | **99.82%** | Hypercube | 100% |
| **Max QPS** | **IVFFlat** | **14,806** | Hypercube | 88 |
| **Best AF** | Neural LSH, LSH | 1.000 | IVFFlat | 1.000 |
| **Best Balance** | **IVFFlat** | 97.8% @ 14.8k | Hypercube | 100% @ 88 |

### 4.2 Overall Performance - SIFT

#### Recall vs QPS Trade-off

```
                    Recall@10
                        ↑
                        |
    100% ├────●─────────────────────
         |   LSH
    95%  ├───────────────●──────────
         |            IVFFlat
    85%  ├────────────────────●─────
         |                 Hyper
    74%  ├────────────────────────●─
         |                    Neural
    52%  ├────────────────────●─────
         |                 IVFPQ
     0%  └──────────────────────────→ QPS
         0   2.3 2.9 29  2.5k  7k 8.6k
```

**Κατηγοριοποίηση:**
- **High Recall, Low Speed:** LSH, IVFFlat, Hypercube
- **Balanced:** Neural LSH
- **High Speed, Low Recall:** IVFPQ

#### Winner per Metric - SIFT

| Μετρική | Νικητής | Τιμή | 2ος | Τιμή |
|---------|---------|------|-----|------|
| **Max Recall** | **LSH** | **100%** | IVFFlat | 99.4% |
| **Max QPS** | **IVFPQ** | **8,610** | IVFFlat | 7,025 |
| **Best Speedup** | **IVFPQ** | **8.61×** | IVFFlat | 7.87× |
| **Best Balance** | **IVFFlat** | **95.9% @ 7k** | Neural LSH | 74% @ 29 |

### 4.3 Comparative Summary Table

#### MNIST - Best Configurations

| Αλγόριθμος | Recall@10 | QPS | t_approx | Speedup | Κατάλληλο για |
|------------|-----------|-----|----------|---------|---------------|
| **Neural LSH** | 99.82% | 15 | 66ms | - | High-accuracy ML |
| **LSH** | 99.4% | 27 | 37ms | 1.5× | Balanced apps |
| **Hypercube** | 100% | 88 | 11ms | - | Real-time + accuracy |
| **IVFFlat** | 97.8% | 14.8k | 0.07ms | 2.8× | Production (best!) |
| **IVFPQ** | 77.2% | 4.7k | 0.21ms | 4.7× | Memory-constrained |

#### SIFT - Best Configurations

| Αλγόριθμος | Recall@10 | QPS | t_approx | Speedup | Κατάλληλο για |
|------------|-----------|-----|----------|---------|---------------|
| **Neural LSH** | 74.3% | 29 | 34ms | - | High-accuracy needed |
| **LSH** | 97.0% | 3 | 342ms | - | Offline processing |
| **Hypercube** | 85.0% | 3 | 393ms | - | Offline processing |
| **IVFFlat** | 95.9% | 7.0k | 0.14ms | 7.9× | **Production (best!)** |
| **IVFPQ** | 52.1% | 8.6k | 0.12ms | 8.6× | Fast but inaccurate |

---

## 5. Ανάλυση Hyperparameters - Neural LSH

### 5.1 Επίδραση Κρίσιμων Παραμέτρων

#### Effect of m (Number of Partitions)

| m | MNIST Recall | SIFT Recall | Παρατηρήσεις |
|---|--------------|-------------|--------------|
| 100 | 95.4% | 4.8% | Πολλά, μικρά bins - χαμηλό recall |
| 50 | 99.1% | 20.9% | Μέτρια bins |
| 30 | 99.6% | 39.7% | Καλή ισορροπία |
| 25 | **99.8%** | 61.6% | Λίγα, μεγάλα bins - υψηλό recall |
| **15** | - | **74.3%** | **Βέλτιστο για SIFT** |

**Συμπέρασμα:** 
- Μικρότερο m → μεγαλύτερα partitions → υψηλότερο recall
- SIFT χρειάζεται πολύ μικρό m (15-25) για >70% recall
- MNIST πιο εύκολο: m=25 αρκετό για 99.8%

#### Effect of knn (Graph Richness)

| knn | SIFT Recall | Training Impact |
|-----|-------------|-----------------|
| 10 | 4.8% | Αραιός γράφος, κακό partitioning |
| 20 | 43.4% | Βελτίωση +800% |
| 30 | 61.6% | Πλούσιος γράφος |
| **40** | **74.3%** | **Πολύ πλούσιος - best** |

**Συμπέρασμα:**
- Περισσότεροι γείτονες → καλύτερη ποιότητα partitioning
- knn=40 απαραίτητο για SIFT high recall

#### Effect of epochs & learning rate

| Config | epochs | lr | SIFT Recall | Training Time |
|--------|--------|-----|-------------|---------------|
| Fast | 10 | 0.001 | 4.8% | ~2 min |
| Medium | 30 | 0.0005 | 39.7% | ~5 min |
| **Optimal** | **70** | **0.00015** | **74.3%** | **~10 min** |

**Συμπέρασμα:**
- Χαμηλό lr (0.00015) + πολλά epochs → σταθερή σύγκλιση
- SIFT δύσκολο → χρειάζεται προσεκτική εκμάθηση

#### Effect of T (Multi-Probe)

| T | % of bins probed | SIFT Recall | QPS |
|---|------------------|-------------|-----|
| 5 | 33% (m=15) | 71.2% | 35 |
| **10** | **67%** | **74.3%** | **29** |
| 15 | 100% (m=15) | 75.1% | 22 |

**Συμπέρασμα:**
- T=10 (67% των bins) βέλτιστο
- Περισσότερα probes → μικρή βελτίωση, μεγάλη επιβάρυνση

### 5.2 Learning Rate & Batch Size Analysis

#### Impact on Convergence

| lr | batch_size | SIFT Recall | Convergence |
|----|------------|-------------|-------------|
| 0.01 | 256 | 35% | Ασταθής, oscillates |
|