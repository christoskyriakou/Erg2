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

## 3. Περιγραφή Αλγορίθμου Neural LSH

Ο Neural LSH είναι μια μέθοδος προσεγγιστικής αναζήτησης που χωρίζει τον χώρο των δεδομένων σε περιοχές (partitions) με τρόπο που διατηρεί τη γεωμετρική τοπικότητα.  
Αποτελείται από τα παρακάτω στάδια:

### 3.1 Κατασκευή k-NN γράφου
Για κάθε διάνυσμα βρίσκουμε τους k κοντινότερους γείτονες (με L2).  
Ο γράφος αυτός χρησιμοποιείται ως βάση για τον διαμερισμό.

### 3.2 Συμμετρικοποίηση και βάρη
- Αν δύο σημεία είναι αμοιβαίοι γείτονες → βάρος 2  
- Αν είναι μονόπλευροι → βάρος 1  

### 3.3 Μετατροπή σε CSR για KaHIP
Ο γράφος μετατρέπεται στους πίνακες:
- `xadj`
- `adjncy`
- `adjcwgt`
- `vwgt`

ώστε να μπορεί να χρησιμοποιηθεί από το KaHIP.

### 3.4 Διαμέριση με KaHIP
Το KaHIP χωρίζει τον γράφο σε *m* partitions με ισορροπία.  
Κάθε σημείο παίρνει ένα label (το partition στο οποίο ανήκει).

### 3.5 Εκπαίδευση MLP
Με βάση τα labels του KaHIP εκπαιδεύεται ένα MLP που μαθαίνει να προβλέπει  
σε ποιο partition ανήκει ένα νέο σημείο.

### 3.6 Inverted File Index
Μετά την πρόβλεψη:
- Δημιουργείται ένας inverted index:  
  **partition → λίστα από point IDs**

### 3.7 Multi-Probe Search
Κατά την αναζήτηση:
1. Το MLP προβλέπει πιθανότητες για όλα τα bins.  
2. Επιλέγονται τα **T** καλύτερα bins.  
3. Συλλέγονται οι υποψήφιοι από αυτά τα bins.  
4. Υπολογίζονται ακριβείς L2 αποστάσεις στους υποψηφίους.  
5. Επιστρέφονται οι Ν κοντινότεροι ή τα R-near neighbors.

## 4. Χρήση Scripts

Η λειτουργικότητα του Neural LSH βασίζεται σε δύο κύρια scripts:

- **`nlsh_build.py`** → Κατασκευή του Neural LSH index  
- **`nlsh_search.py`** → Αναζήτηση (k-NN ή range search) μέσα από το index  

---

## 4.1 Κατασκευή Ευρετηρίου — `nlsh_build.py` 🧠

Το script αυτό:
- δημιουργεί τον k-NN γράφο,
- εκτελεί το partitioning με KaHIP,
- εκπαιδεύει το MLP,
- και παράγει το τελικό NLSH index.

---

### 🛠️ Συντακτικό

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

| Παράμετρος      | Περιγραφή                                 | Default |
|-----------------|-------------------------------------------|---------|
| `-d`            | Dataset αρχείο (π.χ. `res/sift_base.fvecs`) | — |
| `-i`            | Φάκελος/πρόθεμα για αποθήκευση του index  | — |
| `-type`         | Τύπος δεδομένων: `mnist` ή `sift`          | — |
| `--knn`         | k για τον k-NN γράφο                      | 10 |
| `-m`            | Αριθμός partitions (bins)                 | 100 |
| `--imbalance`   | Επιτρεπτή ανισορροπία στο partitioning    | 0.03 |
| `--kahip_mode`  | 0 = FAST, 1 = ECO, 2 = STRONG             | 2 |
| `--layers`      | Αριθμός κρυφών στρωμάτων του MLP          | 3 |
| `--nodes`       | Νευρώνες ανά κρυφό layer                  | 64 |
| `--epochs`      | Εποχές εκπαίδευσης                        | 10 |
| `--batch_size`  | Batch size                                | 128 |
| `--lr`          | Learning rate                             | 0.001 |
| `--seed`        | Random seed                               | 1 |
