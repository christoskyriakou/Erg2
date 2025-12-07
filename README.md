# Προγραμματιστική Εργασία 2 – Neural LSH
Κ23γ – Ανάπτυξη Λογισμικού για Αλγοριθμικά Προβλήματα  
Χειμερινό εξάμηνο 2025–26

Η εργασία αυτή υλοποιεί τον αλγόριθμο **Neural LSH** για προσεγγιστική αναζήτηση κοντινότερων γειτόνων σε υψηλοδιάστατα διανύσματα, χρησιμοποιώντας τα datasets MNIST και SIFT.  
Η μέθοδος βασίζεται στα στάδια: κατασκευή k-NN γράφου, ισοκατανεμημένη διαμέριση με **KaHIP**, εκπαίδευση **MLP** και multi-probe αναζήτηση.

---

## 1. Δομή Αποθετηρίου

```text
project2/
├── ivfflat/                 # Κώδικας από Εργασία 1 (IVFFlat) για κατασκευή k-NN γράφου
├── peiramatikes/            # Πειραματική μελέτη
├── BUILD_OUTPUT.pdf
├── README.md                # Το παρόν αρχείο
├── dataset_parser.py        # Parser για MNIST / SIFT
├── distances.py             # L2 distance & exhaustive search
├── graph_utils.py           # k-NN graph, συμμετρικοποίηση, βάρη, CSR για KaHIP
├── models.py                # MLP ταξινομητής (PyTorch)
├── nlsh_build.py            # Κατασκευή ευρετηρίου Neural LSH
├── nlsh_search.py           # Multi-probe αναζήτηση Neural LSH
├── peiramatiki_meleti.md    # Συγκριτική μελέτη (Εργασία 1 + Neural LSH)
└── requirments.txt          # Python dependencies
```
## 2. Εγκατάσταση

### Προαπαιτούμενα

- Linux/Unix περιβάλλον  
- Python 3.10+  
- PyTorch  
- NumPy  
- kahip  

### Εγκατάσταση μέσω `requirments.txt`

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirments.txt
## 3. Περιγραφή Αλγορίθμου Neural LSH

1. **Κατασκευή k-NN γράφου** πάνω στο dataset (MNIST ή SIFT).

2. **Συμμετρικοποίηση & βάρη:**
   - 2 για αμοιβαίους γείτονες  
   - 1 για μονόπλευρους  

3. **CSR format για KaHIP**, όπου δημιουργούνται οι πίνακες:
   - `xadj`  
   - `adjncy`  
   - `adjcwgt`  
   - `vwgt`  

4. **Ισοκατανεμημένη διαμέριση** με `kahip.kaffpa` για `m` partitions.

5. **Εκπαίδευση MLP ταξινομητή** που προβλέπει την partition ετικέτα για κάθε διάνυσμα.

6. **Inverted file index**
   - Αντιστοίχιση `label → point IDs`.

7. **Multi-probe search**
   - Επιλογή των `T` bins με τη μεγαλύτερη πιθανότητα.  
   - Συλλογή όλων των υποψηφίων από αυτά τα bins.  
   - Ακριβής L2 αναζήτηση μέσα στους υποψηφίους για εύρεση των Ν πλησιέστερων.
## 4. Χρήση Scripts

Η λειτουργικότητα του project βασίζεται σε δύο κύρια Python scripts: `nlsh_build.py` για την κατασκευή του ευρετηρίου και `nlsh_search.py` για την εκτέλεση αναζητήσεων.

---

### 4.1 Κατασκευή Ευρετηρίου — `nlsh_build.py` 🧠

Χρησιμοποιείται για την εκπαίδευση του MLP και τη δημιουργία του NLSH index.

#### 🛠️ Συντακτικό

```bash
python nlsh_build.py \
  -d <input file> \
  -i <index path> \
  -type <sift|mnist> \
  --knn <int> \
  -m <int> \
  --imbalance <float> \
  --kahip_mode <int> \
  --layers <int> \
  --nodes <int> \
  --epochs <int> \
  --batch_size <int> \
  --lr <float> \
  --seed <int>
## ⚙️ Παράμετροι Script `nlsh_build.py` (Κατασκευή Ευρετηρίου)

| Παράμετρος | Περιγραφή | Default |
| :--- | :--- | :--- |
| **`-d`** | Dataset αρχείο (π.χ. res/sift_base.fvecs) | — |
| **`-i`** | Φάκελος/πρόθεμα για αποθήκευση του index | — |
| **`-type`** | Τύπος δεδομένων: mnist ή sift | — |
| **`--knn`** | $k$ για τον $k$-NN γράφο (Χρησιμοποιείται στην εκπαίδευση) | 10 |
| **`-m`** | Αριθμός partitions (bins) για KaHIP | 100 |
| **`--imbalance`** | Επιτρεπτή ανισορροπία στον διαμερισμό (KaHIP) | 0.03 |
| **`--kahip_mode`** | Λειτουργία KaHIP: 0 = FAST, 1 = ECO, 2 = STRONG | 2 |
| **`--layers`** | Αριθμός κρυφών στρωμάτων MLP | 3 |
| **`--nodes`** | Νευρώνες ανά στρώμα | 64 |
| **`--epochs`** | Εποχές εκπαίδευσης | 10 |
| **`--batch_size`** | Batch size | 128 |
| **`--lr`** | Learning rate | 0.001 |
| **`--seed`** | Random seed | 1 |
