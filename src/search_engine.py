"""
Search Engine
Μηχανή αναζήτησης με Neural LSH
"""

import numpy as np
import torch


class SearchEngine:
    """Μηχανή αναζήτησης πλησιέστερων γειτόνων"""
    
    def __init__(self, index_manager, data, device='cpu'):
        """
        Args:
            index_manager: IndexManager instance
            data: numpy array με το dataset
            device: PyTorch device
        """
        self.index_manager = index_manager
        self.data = data
        self.device = device
        
        # Μεταφορά μοντέλου στο device
        self.index_manager.model.to(device)
        self.index_manager.model.eval()
    
    def search(self, query, N=1, T=5):
        """
        Αναζήτηση N πλησιέστερων γειτόνων με multi-probe
        
        Args:
            query: numpy array (dim,)
            N: αριθμός πλησιέστερων γειτόνων
            T: αριθμός partitions προς έλεγχο
        
        Returns:
            neighbors: λίστα με indices γειτόνων
            distances: λίστα με αποστάσεις
        """
        # Βήμα 1: Πρόβλεψη πιθανοτήτων με το μοντέλο
        query_tensor = torch.FloatTensor(query).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            probs = self.index_manager.model.predict_proba(query_tensor)
            probs = probs.cpu().numpy()[0]
        
        # Βήμα 2: Επιλογή top-T partitions
        top_partitions = np.argsort(probs)[-T:][::-1]
        
        # Βήμα 3: Συλλογή υποψηφίων σημείων
        candidate_indices = []
        for partition_id in top_partitions:
            points = self.index_manager.get_partition_points(int(partition_id))
            candidate_indices.extend(points)
        
        # Αφαίρεση διπλότυπων
        candidate_indices = list(set(candidate_indices))
        
        if len(candidate_indices) == 0:
            # Fallback: επιστροφή τυχαίων σημείων
            candidate_indices = list(range(min(N, len(self.data))))
        
        # Βήμα 4: Ακριβής αναζήτηση στους υποψηφίους
        candidates = self.data[candidate_indices]
        distances = np.linalg.norm(candidates - query, axis=1)
        
        # Ταξινόμηση και επιλογή top-N
        sorted_indices = np.argsort(distances)[:N]
        
        neighbors = [candidate_indices[i] for i in sorted_indices]
        distances = distances[sorted_indices]
        
        return neighbors, distances
    
    def range_search(self, query, radius, T=5):
        """
        Range search: βρες όλα τα σημεία σε ακτίνα R
        
        Args:
            query: numpy array (dim,)
            radius: ακτίνα αναζήτησης
            T: αριθμός partitions προς έλεγχο
        
        Returns:
            neighbors: λίστα με indices σημείων εντός ακτίνας
        """
        # Πρόβλεψη πιθανοτήτων
        query_tensor = torch.FloatTensor(query).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            probs = self.index_manager.model.predict_proba(query_tensor)
            probs = probs.cpu().numpy()[0]
        
        # Επιλογή top-T partitions
        top_partitions = np.argsort(probs)[-T:][::-1]
        
        # Συλλογή υποψηφίων
        candidate_indices = []
        for partition_id in top_partitions:
            points = self.index_manager.get_partition_points(int(partition_id))
            candidate_indices.extend(points)
        
        candidate_indices = list(set(candidate_indices))
        
        if len(candidate_indices) == 0:
            return []
        
        # Έλεγχος αποστάσεων
        candidates = self.data[candidate_indices]
        distances = np.linalg.norm(candidates - query, axis=1)
        
        # Επιλογή σημείων εντός ακτίνας
        within_radius = distances <= radius
        neighbors = [candidate_indices[i] for i, flag in enumerate(within_radius) if flag]
        
        return neighbors
    
    def true_search(self, query, N=1):
        """
        Εξαντλητική αναζήτηση (ground truth)
        
        Args:
            query: numpy array (dim,)
            N: αριθμός πλησιέστερων γειτόνων
        
        Returns:
            neighbors: λίστα με indices γειτόνων
            distances: λίστα με αποστάσεις
        """
        # Υπολογισμός όλων των αποστάσεων
        distances = np.linalg.norm(self.data - query, axis=1)
        
        # Ταξινόμηση και επιλογή top-N
        sorted_indices = np.argsort(distances)[:N]
        
        neighbors = sorted_indices.tolist()
        distances = distances[sorted_indices]
        
        return neighbors, distances