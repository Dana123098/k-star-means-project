import sys
import os
import time
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import get_iris
from src.kstar_means import kstar_means
from src.plots import plot_metric_vs_iteration, plot_clusters

def main():
   
    os.makedirs("results/graphs", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)

    print("\n" + "=" * 50)
    print("Running K*-Means on REAL DATASET: IRIS")
    print("=" * 50)

    X, y_true = get_iris()
    X_arr = np.array(X)

    start_time = time.time()
    centroids, assignments, final_sse, final_mdl, history = kstar_means(
        X_arr,
        max_iter=50,
        m=32,
        patience=5
    )
    runtime = time.time() - start_time

    detected_k = len(centroids)

    print(f"Execution finished. True classes: 3 | Detected k: {detected_k}")
    print(f"Final SSE: {final_sse:.2f}")
    print(f"Final MDL: {final_mdl:.2f}")
    print(f"Runtime: {runtime:.4f} sec")

    history_path = "results/tables/iris_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    print(f"Saved iteration history to: {history_path}")

    iris_summary = [{
        "dataset": "Iris",
        "true_k": 3,
        "detected_k": detected_k,
        "final_sse": round(final_sse, 2),
        "final_mdl": round(final_mdl, 2),
        "runtime_sec": round(runtime, 4),
        "iterations": len(history)
    }]
    summary_path = "results/tables/iris_summary.csv"
    pd.DataFrame(iris_summary).to_csv(summary_path, index=False)
    print(f"Saved summary table to: {summary_path}")

    plot_metric_vs_iteration(history, "mdl", "results/graphs/mdl_vs_iteration_iris.png")
    plot_metric_vs_iteration(history, "sse", "results/graphs/sse_vs_iteration_iris.png")
    plot_metric_vs_iteration(history, "k", "results/graphs/k_vs_iteration_iris.png")

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_arr)
    centroids_pca = pca.transform(centroids)

    plot_clusters(
        X_pca,
        assignments,
        centroids_pca,
        save_path="results/graphs/clusters_kstar_iris.png"
    )
    print("Saved all Iris plots to results/graphs/")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    main()