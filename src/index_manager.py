"""
Index Manager
Διαχείριση και αποθήκευση ευρετηρίου Neural LSH
"""

import pickle
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict


class IndexManager:
    """Διαχειρίζεται το ευρετήριο Neural LSH"""
    
    def __init__(self):
        self.model = None
        self.inverted_file = None  # Dictionary: partition_id -> list of point indices
        self.n_parts = 0
        self.metadata = {}
    
    def build_index(self, data, partition_labels, model, args):
        """
        Κατασκευή ευρετηρίου
        
        Args:
            data: numpy array με τα δεδομένα
            partition_labels: numpy array με partition labels
            model: εκπαιδευμένο PyTorch μοντέλο
            args: arguments από command line
        """
        self.model = model
        self.n_parts = args.parts
        
        # Κατασκευή inverted file
        # partition_id -> λίστα με indices σημείων που ανήκουν σε αυτό το partition
        print(f"    Building inverted file...")
        self.inverted_file = defaultdict(list)
        
        for point_idx, partition_id in enumerate(partition_labels):
            self.inverted_file[int(partition_id)].append(point_idx)
        
        # Μετατροπή σε κανονικό dictionary
        self.inverted_file = dict(self.inverted_file)
        
        # Αποθήκευση metadata
        self.metadata = {
            'n_samples': len(data),
            'dim': data.shape[1],
            'n_parts': self.n_parts,
            'knn': args.knn,
            'dataset_type': args.type,
            'model_params': {
                'input_dim': model.input_dim,
                'hidden_dim': model.hidden_dim,
                'output_dim': model.output_dim,
                'n_layers': model.n_layers
            }
        }
        
        # Στατιστικά
        sizes = [len(points) for points in self.inverted_file.values()]
        print(f"    Inverted file: {len(self.inverted_file)} partitions")
        print(f"    Partition sizes: min={min(sizes)}, max={max(sizes)}, "
              f"avg={np.mean(sizes):.1f}")
    
    def save(self, index_path):
        """
        Αποθήκευση ευρετηρίου σε αρχείο
        
        Args:
            index_path: path για αποθήκευση (χωρίς extension)
        """
        index_path = Path(index_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Αποθήκευση PyTorch μοντέλου
        model_path = str(index_path) + '_model.pth'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_params': self.metadata['model_params']
        }, model_path)
        
        # Αποθήκευση inverted file και metadata
        index_data = {
            'inverted_file': self.inverted_file,
            'metadata': self.metadata
        }
        
        data_path = str(index_path) + '_data.pkl'
        with open(data_path, 'wb') as f:
            pickle.dump(index_data, f)
        
        print(f"    Saved model to {model_path}")
        print(f"    Saved index to {data_path}")
    
    def load(self, index_path):
        """
        Φόρτωση ευρετηρίου από αρχείο
        
        Args:
            index_path: path του ευρετηρίου (χωρίς extension)
        """
        index_path = Path(index_path)
        
        # Φόρτωση metadata και inverted file
        data_path = str(index_path) + '_data.pkl'
        with open(data_path, 'rb') as f:
            index_data = pickle.load(f)
        
        self.inverted_file = index_data['inverted_file']
        self.metadata = index_data['metadata']
        self.n_parts = self.metadata['n_parts']
        
        # Φόρτωση PyTorch μοντέλου
        model_path = str(index_path) + '_model.pth'
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Επαναδημιουργία μοντέλου
        from src.models import MLPClassifier
        params = checkpoint['model_params']
        
        self.model = MLPClassifier(
            input_dim=params['input_dim'],
            hidden_dim=params['hidden_dim'],
            output_dim=params['output_dim'],
            n_layers=params['n_layers']
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
    
    def get_partition_points(self, partition_id):
        """
        Επιστροφή των indices των σημείων που ανήκουν σε ένα partition
        
        Args:
            partition_id: ID του partition
        
        Returns:
            Λίστα με indices σημείων
        """
        return self.inverted_file.get(partition_id, [])