from sklearn.datasets import make_blobs, load_iris

def get_synthetic():
    X, _ = make_blobs(n_samples=500, centers=4, random_state=42)
    return X.tolist()

def get_iris():
    data = load_iris()
    return data.data.tolist()