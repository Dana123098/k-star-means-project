import sys
import os
import time  
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import get_synthetic_cases
from src.kstar_means import kstar_means
from src.plots import plot_metric_vs_iteration, plot_clusters


def main():
    
    os.makedirs("results/graphs", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)

    cases = get_synthetic_cases()
    all_cases_summary = []

    for case_name, (X, y_true) in cases.items():
        print("\n" + "=" * 40)
        print(f"Running K*-Means for scenario: {case_name.upper()}")
        print("=" * 40)

        X = np.asarray(X)

        start_time = time.time()

        centroids, assignments, final_sse, final_mdl, history = kstar_means(
            X,
            max_iter=50,
            m=32,
            patience=5
        )

        runtime = time.time() - start_time

        detected_k = len(centroids)

        print(f"Detected k: {detected_k}")
        print(f"Final SSE: {final_sse:.2f}")
        print(f"Final MDL: {final_mdl:.2f}")
        print(f"Runtime: {runtime:.4f} sec")  

        history_path = f"results/tables/synthetic_history_{case_name}.csv"
        pd.DataFrame(history).to_csv(history_path, index=False)

        all_cases_summary.append({
            "scenario": case_name,
            "true_k": 4,
            "detected_k": detected_k,
            "final_sse": round(final_sse, 2),
            "final_mdl": round(final_mdl, 2),
            "runtime_sec": round(runtime, 4),  
            "iterations": len(history)
        })

        plot_metric_vs_iteration(
            history,
            metric_name="mdl",
            save_path=f"results/graphs/mdl_vs_iteration_{case_name}.png"
        )

        plot_metric_vs_iteration(
            history,
            metric_name="sse",
            save_path=f"results/graphs/sse_vs_iteration_{case_name}.png"
        )

        plot_metric_vs_iteration(
            history,
            metric_name="k",
            save_path=f"results/graphs/k_vs_iteration_{case_name}.png"
        )

        plot_clusters(
            X,
            assignments,
            centroids,
            save_path=f"results/graphs/clusters_kstar_{case_name}.png"
        )

        print(f"Saved history and plots for: {case_name}")

    summary_path = "results/tables/synthetic_summary.csv"
    pd.DataFrame(all_cases_summary).to_csv(summary_path, index=False)

    print("\n" + "=" * 40)
    print("All synthetic experiments completed.")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()