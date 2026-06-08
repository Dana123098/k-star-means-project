import numpy as np
from src.kmeans import kmeans
from src.mdl import calculate_mdl, calculate_sse

def assign_points(points, centroids):
  
    distances = np.linalg.norm(
        points[:, np.newaxis, :] - centroids[np.newaxis, :, :],
        axis=2
    )
    return np.argmin(distances, axis=1)

# FUNKCJA PODZIAŁU 
#( Порог для сплита взят строго из математического вывода статьи (Секция 3.2.1).)
def maybe_split(points, centroids, assignments):
    """
    
    """
    n, d = points.shape
    k = len(centroids)

    for i in range(k):
        cluster_mask = assignments == i
        cluster_data = points[cluster_mask]

        if len(cluster_data) < 4:
            continue

        sse_parent = np.sum((cluster_data - centroids[i]) ** 2)

        sub_centroids, sub_assignments, sse_sub = kmeans(
            cluster_data,
            k=2,
            max_iter=20
        )

        
        if sse_parent - sse_sub > (4 * n) / (k + 1):
            centroids = np.delete(centroids, i, axis=0)
            centroids = np.vstack([centroids, sub_centroids])
            return centroids, True

    return centroids, False

# FUNKCJA POŁĄCZENIA 
def maybe_merge(points, centroids, assignments):
    
    k = len(centroids)

    if k <= 1:
        return centroids, False

    n, d = points.shape
    distances = np.linalg.norm(
        centroids[:, np.newaxis, :] - centroids[np.newaxis, :, :],
        axis=2
    )
    np.fill_diagonal(distances, np.inf)
    i1, i2 = np.unravel_index(np.argmin(distances), distances.shape)

    cluster1_data = points[assignments == i1]
    cluster2_data = points[assignments == i2]

    if len(cluster1_data) == 0 or len(cluster2_data) == 0:
        return centroids, False

    merged_data = np.vstack([cluster1_data, cluster2_data])

    sse_1 = np.sum((cluster1_data - centroids[i1]) ** 2)
    sse_2 = np.sum((cluster2_data - centroids[i2]) ** 2)
    merged_centroid = np.mean(merged_data, axis=0)
    sse_merged = np.sum((merged_data - merged_centroid) ** 2)

    if sse_merged - (sse_1 + sse_2) < (4 * n) / k:
        centroids = np.delete(centroids, [i1, i2], axis=0)
        centroids = np.vstack([centroids, merged_centroid])
        return centroids, True

    return centroids, False


def kstar_means(points, max_iter=50, m=32, patience=5):
    
    points = np.asarray(points)
    n, d = points.shape

    centroids = np.array([np.mean(points, axis=0)])
    assignments = np.zeros(n, dtype=int)

    history = []
    best_mdl = np.inf
    unimproved_count = 0

    for iteration in range(max_iter):
        assignments = assign_points(points, centroids)
        for i in range(len(centroids)):
            cluster_points = points[assignments == i]
            if len(cluster_points) > 0:
                centroids[i] = np.mean(cluster_points, axis=0)

        did_merge = False
        centroids, did_split = maybe_split(points, centroids, assignments)

        if not did_split:
            centroids, did_merge = maybe_merge(points, centroids, assignments)

        
        assignments = assign_points(points, centroids)
        sse = calculate_sse(points, centroids, assignments)
        mdl = calculate_mdl(points, centroids, assignments, m)

        operation = "split" if did_split else ("merge" if did_merge else "none")

        history.append({
            "iteration": iteration,
            "k": len(centroids),
            "sse": sse,
            "mdl": mdl,
            "operation": operation
        })

        if mdl < best_mdl:
            best_mdl = mdl
            unimproved_count = 0
        else:
            unimproved_count += 1

        if unimproved_count >= patience:
            break

    assignments = assign_points(points, centroids)
    final_sse = calculate_sse(points, centroids, assignments)
    final_mdl = calculate_mdl(points, centroids, assignments, m)

    return centroids, assignments, final_sse, final_mdl, history