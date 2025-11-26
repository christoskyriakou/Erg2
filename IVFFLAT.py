import numpy as np
import struct
import os
from ivfflat_utils import (closest_centroid,l2,topn_probe,kmeans,)
class IVFFLAT:
    def __init__(self,d,nlist):
        self.d=d
        self.nlist=nlist
        self.centroids=None
        self.invlists=[[]for _ in range(nlist)]
        self.X = None

    def train(self,X):
        self.centroids=kmeans(X,self.nlist)
        self.X=X

    def add(self,X):
        for i,q in enumerate(X):
            cid=closest_centroid(self.centroids,q)
            self.invlists[cid].append(i)

    def search(self,q,k,nprobe):
        top_centroids=topn_probe(self.centroids,q,nprobe)
        candidates=[]
        for cids in top_centroids:
            for index in self.invlists[cids]:
                cand_vector=self.X[index]
                dist=l2(cand_vector,q)
                candidates.append((dist, index))
        candidates.sort()
        return candidates[:k]

           
            
        

