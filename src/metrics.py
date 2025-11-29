"""
Metrics Calculator
Υπολογισμός μετρικών απόδοσης (AF, Recall)
"""

import numpy as np


class MetricsCalculator:
    """Υπολογισμός μετρικών απόδοσης"""
    
    def calculate_af(self, results):
        """
        Υπολογισμός Average Approximation Factor (AF)
        
        AF = (approximate_distance / true_distance) για τον 1ο γείτονα
        
        Args:
            results: λίστα από dictionaries με αποτελέσματα queries
        
        Returns:
            average AF
        """
        afs = []
        
        for result in results:
            if len(result['distances']) > 0 and len(result['true_distances']) > 0:
                approx_dist = result['distances'][0]
                true_dist = result['true_distances'][0]
                
                if true_dist > 0:
                    af = approx_dist / true_dist
                    afs.append(af)
        
        if len(afs) == 0:
            return 1.0
        
        return np.mean(afs)
    
    def calculate_recall(self, results, N):
        """
        Υπολογισμός Recall@N
        
        Recall@N = (πλήθος true neighbors στα approximate neighbors) / N
        
        Args:
            results: λίστα από dictionaries με αποτελέσματα queries
            N: αριθμός γειτόνων
        
        Returns:
            average recall
        """
        recalls = []
        
        for result in results:
            approx_neighbors = set(result['neighbors'][:N])
            true_neighbors = set(result['true_neighbors'][:N])
            
            # Υπολογισμός intersection
            common = len(approx_neighbors.intersection(true_neighbors))
            recall = common / N if N > 0 else 0
            recalls.append(recall)
        
        if len(recalls) == 0:
            return 0.0
        
        return np.mean(recalls)
    
    def calculate_precision(self, results, N):
        """
        Υπολογισμός Precision@N
        
        Args:
            results: λίστα από dictionaries με αποτελέσματα queries
            N: αριθμός γειτόνων
        
        Returns:
            average precision
        """
        precisions = []
        
        for result in results:
            approx_neighbors = set(result['neighbors'][:N])
            true_neighbors = set(result['true_neighbors'][:N])
            
            if len(approx_neighbors) > 0:
                common = len(approx_neighbors.intersection(true_neighbors))
                precision = common / len(approx_neighbors)
                precisions.append(precision)
        
        if len(precisions) == 0:
            return 0.0
        
        return np.mean(precisions)