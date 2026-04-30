import matplotlib.pyplot as plt


def plot_metric_vs_k(history, metric_name, save_path):
    """
    Строит график зависимости метрики от k.
    Например: SSE vs k, MDL vs k, silhouette vs k.
    """

    ks = [item["k"] for item in history]
    values = [item[metric_name] for item in history]

    plt.figure(figsize=(8, 5))
    plt.plot(ks, values, marker="o")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel(metric_name.upper())
    plt.title(f"{metric_name.upper()} vs k")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_clusters(points, labels, centroids, save_path):
    """
    Строит scatter plot кластеров.
    points - список точек
    labels - номер кластера для каждой точки
    centroids - центры кластеров
    """

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    plt.figure(figsize=(8, 6))
    plt.scatter(xs, ys, c=labels, cmap="tab10", s=35, alpha=0.7)

    centroid_x = [c[0] for c in centroids]
    centroid_y = [c[1] for c in centroids]

    plt.scatter(
        centroid_x,
        centroid_y,
        c="black",
        marker="X",
        s=200,
        label="Centroids"
    )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Clusters selected by K*-Means")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()