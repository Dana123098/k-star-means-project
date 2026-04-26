from kmeans import kmeans, compute_sse
from sklearn.metrics import silhouette_score


def kstar_means(points, k_min=2, k_max=10, method="silhouette"):
    """
    Автоматический выбор числа кластеров k.

    parameters:
        points - список точек [[x, y], ...]
        k_min - минимальное число кластеров
        k_max - максимальное число кластеров
        method - "silhouette" или "sse"

    returns:
        best_clusters
        best_centroids
        best_k
        history (список результатов)
    """

    best_score = None
    best_clusters = None
    best_centroids = None
    best_k = None

    history = []

    for k in range(k_min, k_max + 1):
        clusters, centroids = kmeans(points, k)

        # превращаем кластеры в labels (нужно для silhouette)
        labels = []
        for i, cluster in enumerate(clusters):
            for _ in cluster:
                labels.append(i)

        sse = compute_sse(clusters, centroids)

        if method == "silhouette":
            # silhouette требует минимум 2 кластера
            if len(set(labels)) > 1:
                score = silhouette_score(points, labels)
            else:
                score = -1
        else:
            # если используем SSE → хотим МЕНЬШЕ
            score = -sse

        history.append({
            "k": k,
            "sse": sse,
            "score": score
        })

        if best_score is None or score > best_score:
            best_score = score
            best_clusters = clusters
            best_centroids = centroids
            best_k = k

    return best_clusters, best_centroids, best_k, history