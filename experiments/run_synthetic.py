import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.datasets import make_blobs
from src.kstar_means import kstar_means
from src.plots import plot_metric_vs_k, plot_clusters


os.makedirs("results", exist_ok=True)

X, _ = make_blobs(
    n_samples=500,
    centers=4,
    cluster_std=1.0,
    random_state=42
)

points = X.tolist()

clusters, centroids, labels, best_k, history = kstar_means(
    points,
    k_min=1,
    k_max=10
)

print("Best k:", best_k)

for h in history:
    print(h)

df_results = pd.DataFrame(history)
df_results.to_csv("results/synthetic_results.csv", index=False)
print("Results saved to results/synthetic_results.csv")

plot_metric_vs_k(
    history,
    metric_name="sse",
    save_path="results/sse_vs_k.png"
)

plot_metric_vs_k(
    history,
    metric_name="mdl",
    save_path="results/mdl_vs_k.png"
)

history_for_silhouette = [
    h for h in history if h["silhouette"] is not None
]

plot_metric_vs_k(
    history_for_silhouette,
    metric_name="silhouette",
    save_path="results/silhouette_vs_k.png"
)

plot_clusters(
    points,
    labels,
    centroids,
    save_path="results/clusters_kstar.png"
)

print("Plots saved to results/")