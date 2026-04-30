from src.kmeans import kmeans, compute_sse
from src.mdl import calculate_mdl
from sklearn.metrics import silhouette_score


def kstar_means(points, k_min=1, k_max=10):
    """
    K*-Means with MDL-based model selection.
    The best k is selected by minimizing MDL.
    """

    best_mdl = float("inf")
    best_clusters = None
    best_centroids = None
    best_labels = None
    best_k = None

    history = []

    for k in range(k_min, k_max + 1):
        clusters, centroids, labels = kmeans(points, k)

        sse = compute_sse(clusters, centroids)
        mdl = calculate_mdl(points, centroids, sse)

        if k > 1 and len(set(labels)) > 1:
            silhouette = silhouette_score(points, labels)
        else:
            silhouette = None

        history.append({
            "k": k,
            "sse": sse,
            "mdl": mdl,
            "silhouette": silhouette
        })

        if mdl < best_mdl:
            best_mdl = mdl
            best_clusters = clusters
            best_centroids = centroids
            best_labels = labels
            best_k = k

    return best_clusters, best_centroids, best_labels, best_k, history