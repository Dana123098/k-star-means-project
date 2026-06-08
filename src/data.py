import numpy as np
from sklearn.datasets import make_blobs, load_iris

def get_synthetic(n_samples=500, centers=4, cluster_std=1.0, random_state=42):
    
    X, y = make_blobs(
        n_samples=n_samples,
        centers=centers,
        cluster_std=cluster_std,
        random_state=random_state
    )
    return X, y


def get_synthetic_cases():
  
    return {
        "close": get_synthetic(cluster_std=2.5),
        "medium": get_synthetic(cluster_std=1.2),
        "far": get_synthetic(cluster_std=0.5)
    }


def get_iris():
    
    data = load_iris()
    return data.data, data.target