import sys
import os
import time
import warnings
import pandas as pd
import numpy as np

np.warnings = warnings

from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import get_synthetic_cases
from src.kmeans import kmeans
from src.kstar_means import kstar_means
from src.plots import plot_method_comparison

def safe_silhouette(X, labels):
    unique_labels = set(labels)

    if -1 in unique_labels:
        unique_labels.remove(-1)

    if len(unique_labels) < 2:
        return None

    try:
        return silhouette_score(X, labels)
    except Exception:
        return None

def count_clusters(labels):
    unique_labels = set(labels)
    return len(unique_labels) - (1 if -1 in unique_labels else 0)

def run_manual_kmeans(X, k=4):
    start = time.time()
    centroids, labels, sse = kmeans(X, k=k)
    runtime = time.time() - start

    sil = safe_silhouette(X, labels)

    return {
        "Method": "Manual K-Means",
        "Selected K": k,
        "Criterion": "Given manually",
        "SSE": round(sse, 2),
        "Silhouette": round(sil, 4) if sil is not None else "N/A",
        "MDL": "N/A",
        "Runtime_sec": round(runtime, 4),
        "Notes": "Requires known k"
    }

def run_dbscan(X):
    start = time.time()
    model = DBSCAN()
    labels = model.fit_predict(X)
    runtime = time.time() - start

    selected_k = count_clusters(labels)
    noise_points = int(np.sum(labels == -1))
    sil = safe_silhouette(X, labels)

    return {
        "Method": "DBSCAN",
        "Selected K": selected_k,
        "Criterion": "Default eps/min_samples",
        "SSE": "N/A",
        "Silhouette": round(sil, 4) if sil is not None else "N/A",
        "MDL": "N/A",
        "Runtime_sec": round(runtime, 4),
        "Notes": f"Noise points: {noise_points}"
    }

def run_xmeans(X):
    try:
        from pyclustering.cluster.xmeans import xmeans
        from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
    except ImportError:
        return {
            "Method": "X-Means",
            "Selected K": "N/A",
            "Criterion": "BIC",
            "SSE": "N/A",
            "Silhouette": "N/A",
            "MDL": "N/A",
            "Runtime_sec": "N/A",
            "Notes": "pyclustering not installed"
        }

    start = time.time()

    data = X.tolist()
    initial_centers = kmeans_plusplus_initializer(data, 2).initialize()

    model = xmeans(
        data=data,
        initial_centers=initial_centers,
        kmax=20
    )

    model.process()
    clusters = model.get_clusters()

    labels = np.empty(len(X), dtype=int)
    for cluster_id, cluster_indices in enumerate(clusters):
        labels[cluster_indices] = cluster_id

    runtime = time.time() - start

    selected_k = len(clusters)
    sil = safe_silhouette(X, labels)

    return {
        "Method": "X-Means",
        "Selected K": selected_k,
        "Criterion": "BIC with kmax=20",
        "SSE": "N/A",
        "Silhouette": round(sil, 4) if sil is not None else "N/A",
        "MDL": "N/A",
        "Runtime_sec": round(runtime, 4),
        "Notes": "Requires kmax"
    }

def run_kstar_means(X):
    start = time.time()
    centroids, labels, sse, mdl, history = kstar_means(X)
    runtime = time.time() - start

    selected_k = len(centroids)
    sil = safe_silhouette(X, labels)

    return {
        "Method": "K*-Means",
        "Selected K": selected_k,
        "Criterion": "MDL single adaptive run",
        "SSE": round(sse, 2),
        "Silhouette": round(sil, 4) if sil is not None else "N/A",
        "MDL": round(mdl, 2),
        "Runtime_sec": round(runtime, 4),
        "Notes": "Parameter-free"
    }

def main():
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/graphs", exist_ok=True)

    cases = get_synthetic_cases()
    X, y_true = cases["medium"]
    X = np.asarray(X)

    print("=" * 60)
    print("Running method comparison on MEDIUM synthetic dataset")
    print("=" * 60)

    results = [
        run_manual_kmeans(X, k=4),
        run_dbscan(X),
        run_xmeans(X),
        run_kstar_means(X)
    ]

    df = pd.DataFrame(results)

    print(df.to_string(index=False))

    table_path = "results/tables/method_comparison.csv"
    df.to_csv(table_path, index=False)

    plot_method_comparison(
        df,
        save_path="results/graphs/method_comparison_k.png"
    )

    print("=" * 60)
    print(f"Saved comparison table to: {table_path}")
    print("Saved comparison plot to: results/graphs/method_comparison_k.png")


if __name__ == "__main__":
    main()