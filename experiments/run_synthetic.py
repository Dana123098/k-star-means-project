from sklearn.datasets import make_blobs
from kstar_means import kstar_means

# генерируем данные
X, _ = make_blobs(n_samples=500, centers=4, random_state=42)

# переводим в список
points = X.tolist()

clusters, centroids, best_k, history = kstar_means(points)

print("Best k:", best_k)

for h in history:
    print(h)