import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.datasets import make_blobs
from src.kmeans import kmeans, compute_sse
from src.kstar_means import kstar_means


os.makedirs("results", exist_ok=True)

X, _ = make_blobs(
    n_samples=500,
    centers=4,
    cluster_std=1.0,
    random_state=42
)

points = X.tolist()

clusters, centroids, labels, best_k_mdl, history = kstar_means(
    points,
    k_min=1,
    k_max=10
)

manual_k = 4

manual_clusters, manual_centroids, manual_labels = kmeans(points, manual_k)
manual_sse = compute_sse(manual_clusters, manual_centroids)

silhouette_candidates = [
    h for h in history if h["silhouette"] is not None
]

best_silhouette_row = max(
    silhouette_candidates,
    key=lambda h: h["silhouette"]
)

best_mdl_row = min(
    history,
    key=lambda h: h["mdl"]
)

comparison = pd.DataFrame([
    {
        "method": "Manual k-means",
        "selected_k": manual_k,
        "criterion": "Given manually",
        "sse": manual_sse,
        "silhouette": None,
        "mdl": None
    },
    {
        "method": "Silhouette selection",
        "selected_k": best_silhouette_row["k"],
        "criterion": "Maximum silhouette",
        "sse": best_silhouette_row["sse"],
        "silhouette": best_silhouette_row["silhouette"],
        "mdl": best_silhouette_row["mdl"]
    },
    {
        "method": "MDL K*-Means",
        "selected_k": best_mdl_row["k"],
        "criterion": "Minimum MDL",
        "sse": best_mdl_row["sse"],
        "silhouette": best_mdl_row["silhouette"],
        "mdl": best_mdl_row["mdl"]
    }
])

print(comparison.to_string(index=False))

comparison.to_csv("results/method_comparison.csv", index=False)

print("Comparison saved to results/method_comparison.csv")