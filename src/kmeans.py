import random
import math

def dist(p, q):
    return sum((pi - qi) ** 2 for pi, qi in zip(p, q))

def center(points):
    return [sum(coords) / len(points) for coords in zip(*points)]

def assign(points, centroids):
    assignments = []
    for p in points:
        dists = [dist(p, c) for c in centroids]
        assignments.append(dists.index(min(dists)))
    return assignments

def kmeans(points, k, max_iter=100, eps=1e-4):
    centroids = random.sample(points, k)

    for _ in range(max_iter):
        assignments = assign(points, centroids)

        clusters = [[] for _ in range(k)]
        for i, p in enumerate(points):
            clusters[assignments[i]].append(p)

        new_centroids = []
        for i, cluster in enumerate(clusters):
            if cluster:
                new_centroids.append(center(cluster))
            else:
                new_centroids.append(random.choice(points))

        shift = max(dist(c1, c2) for c1, c2 in zip(centroids, new_centroids))
        if shift < eps:
            break

        centroids = new_centroids

    return clusters, centroids

def compute_sse(clusters, centroids):
    sse = 0
    for i, cluster in enumerate(clusters):
        for p in cluster:
            sse += dist(p, centroids[i])
    return sse