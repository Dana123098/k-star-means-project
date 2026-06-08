import numpy as np
import matplotlib.pyplot as plt


def plot_metric_vs_iteration(history, metric_name, save_path):
    
    iterations = [item["iteration"] for item in history]
    values = [item[metric_name] for item in history]

    plt.figure(figsize=(8, 5))
    plt.plot(
        iterations,
        values,
        marker="o",
        color="blue",
        linewidth=2,
        label=metric_name.upper()
    )

    ax = plt.gca()

    for item in history:
        operation = item.get("operation")

        if operation in ["split", "merge"]:
            color = "red" if operation == "split" else "green"

            plt.axvline(
                x=item["iteration"],
                color=color,
                linestyle="--",
                alpha=0.6,
                linewidth=1.5
            )
            ax.text(
                item["iteration"],
                0.95,
                f" {operation.upper()}",
                rotation=90,
                color=color,
                transform=ax.get_xaxis_transform(),
                verticalalignment="top",
                fontweight="bold",
                alpha=0.8
            )

    plt.xlabel("Iteration")
    plt.ylabel(metric_name.upper())
    plt.title(f"{metric_name.upper()} vs Iteration")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_clusters(points, labels, centroids, save_path):
   
    points = np.asarray(points)
    centroids = np.asarray(centroids)

    plt.figure(figsize=(8, 6))
    plt.scatter(
        points[:, 0],
        points[:, 1],
        c=labels,
        cmap="tab10",
        s=35,
        alpha=0.7,
        edgecolors="white",
        linewidths=0.5
    )
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        c="black",
        marker="X",
        s=250,
        label="Centroids"
    )

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Clusters found by K*-Means")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_method_comparison(df, save_path):
   
    plt.figure(figsize=(8, 5))

    plt.bar(
        df["Method"],
        df["Selected K"]
    )

    plt.xlabel("Method")
    plt.ylabel("Selected K")
    plt.title("Selected Number of Clusters by Method")
    plt.xticks(rotation=15)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()