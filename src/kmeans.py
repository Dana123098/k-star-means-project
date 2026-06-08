import numpy as np

def kmeans(points, k, max_iter=100, eps=1e-4):
   
    points = np.asarray(points)
    n_samples, n_features = points.shape
    
    # Фиксируем генератор случайных чисел для воспроизводимости результатов
    np.random.seed(42)
    
    # Инициализация: случайным образом выбираем k уникальных точек из датасета
    initial_indices = np.random.choice(n_samples, k, replace=False)
    centroids = points[initial_indices]

    for _ in range(max_iter):
       

        distances = np.linalg.norm(points[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
        
        # Шаг Assign: привязываем каждую точку к индексу ближайшего центроида
        assignments = np.argmin(distances, axis=1)

        # Шаг Update: przeliczenie nowych współrzędnych centroidów jako średnia arytmetyczna punktów klastra.
        new_centroids = np.zeros_like(centroids)
        for i in range(k):
            cluster_points = points[assignments == i]
            if len(cluster_points) > 0:
                new_centroids[i] = cluster_points.mean(axis=0)
            else:
                new_centroids[i] = points[np.random.choice(n_samples)]

        # Проверяем критерий сходимости (сдвинулись ли центроиды)
        shift = np.max(np.sum((centroids - new_centroids) ** 2, axis=1))
        if shift < eps:
            break

        centroids = new_centroids

    # строго пересчитываем метки для итоговых центроидов
    distances = np.linalg.norm(points[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
    assignments = np.argmin(distances, axis=1)
    
    # Вычисляем финальный SSE (Сумма квадратов расстояний до центров)
    sse = 0.0
    for i in range(k):
        cluster_points = points[assignments == i]
        if len(cluster_points) > 0:
            sse += np.sum((cluster_points - centroids[i]) ** 2)

    return centroids, assignments, sse