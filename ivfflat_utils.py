import numpy as np
import struct
import os

def l2(a,b):
    dist=np.sum((a-b)**2)
    return dist

def closest_centroid(centroids,q):
    dist_list=[l2(c,q) for c in centroids]
    centroid_id=np.argmin(dist_list)
    return centroid_id

def topn_probe(centroids,q,nprobe):
    k=nprobe
    dist_list=[l2(c,q) for c in centroids]
    ids_list=np.argsort(dist_list)
    return ids_list[:k]

def kmeans(X, k, niter=20):
    n=X.shape[0]
    d=X.shape[1]
    random_cids=np.random.choice(n,k,replace=False)
    centroids=X[random_cids].copy()
    for _ in range(niter):
        labels=[]
        for q in X:
            dist=[l2(q,c) for c in centroids]
            labels.append(np.argmin(dist))
        labels = np.array(labels)
        new_centroids = np.zeros_like(centroids)
        for cid in range(k):
            centroid_recs= X[labels == cid] # επιστρέφει μοναχα τσ σημεία που ειναι στο συηκεκριμένο centroid
            if len(centroid_recs)>0:
                new_centroids[cid]=np.mean(centroid_recs,axis=0)
            else:
                new_centroids[cid]=centroids[cid] # αν δεν υπαρχουν σημεια για το συγκεκριμενο centroid, το αφηνουμε ως ειναι
        centroids=new_centroids
    return centroids
