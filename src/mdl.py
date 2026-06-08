import numpy as np

def calculate_sse(points, centroids, assignments):
    
    points = np.asarray(points)
    centroids = np.asarray(centroids)
    
    sse = 0.0
    for i in range(len(centroids)):
        cluster_points = points[assignments == i]
        if len(cluster_points) > 0:
            sse += np.sum((cluster_points - centroids[i]) ** 2)
    return sse


def calculate_mdl(points, centroids, assignments, m=32):
    
    points = np.asarray(points)
    centroids = np.asarray(centroids)
    
    n, d = points.shape
    k = len(centroids)
    
    if k <= 0:
        return np.inf
        
    # Wywołanie powyższej funkcji SSE do oceny dopasowania danych.
    sse = calculate_sse(points, centroids, assignments)
    
    # 1. Model Cost: koszt zapisu współrzędnych centroidów (im więcej klastrów, tym wyższy koszt).
    model_cost = k * d * m
    
    # 2. Index Cost: koszt przechowywania etykiet (indeksów) punktów dla przypisanych klastrów.
    index_cost = n * np.log2(k) if k > 1 else 0.0
    
    # 3. Residual Cost: koszt błędów (reszt). Informuje, ile informacji tracimy przez przybliżenie centroidami.
    residual_cost = (n * d * np.log2(2 * np.pi) + sse) / 2
    
    return model_cost + index_cost + residual_cost