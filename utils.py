import numpy as np
from sklearn.cluster import AgglomerativeClustering


def get_hierarchical_clustering(distances, n_clusters):
    clusterer = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='complete')
    cluster_labels = clusterer.fit_predict(distances)
    return cluster_labels

def dataset_augmentation_using_distances(X, distances, augmentation_factor=5, n_clusters=5, n_neighbors=5, interpolation_factor=.1):
    X_total = []

    cluster_labels = get_hierarchical_clustering(distances, n_clusters)
    for cluster in np.unique(cluster_labels):
        cluster_indices = np.where(cluster_labels == cluster)[0]
        n_samples_for_cluster = len(cluster_indices)
        X_cluster = X[cluster_indices]
        if len(X_cluster) < n_neighbors:
            continue
        dm_cluster = distances[cluster_indices, :]
        dm_cluster = dm_cluster[:, cluster_indices]
        np.fill_diagonal(dm_cluster, np.inf)

        knn = np.argsort(dm_cluster, axis=1)[:, :n_neighbors]
        augmentation_indices = np.random.choice(n_samples_for_cluster, augmentation_factor * n_samples_for_cluster, replace=True)
        neighbor_indices = knn[augmentation_indices, np.random.choice(n_neighbors, augmentation_factor * n_samples_for_cluster, replace=True)]
        
        augmentation_samples = X_cluster[augmentation_indices]
        neighbor_samples = X_cluster[neighbor_indices]
        sample_diffs = neighbor_samples - augmentation_samples
        new_samples = neighbor_samples + interpolation_factor * np.random.rand(n_samples_for_cluster * augmentation_factor)[:, np.newaxis] * sample_diffs
        X_total.append(new_samples)
    X_total = np.concatenate(X_total)

    X_total_minima = X_total.min(axis=-1)
    X_needs_shift = X_total_minima < 0
    X_total[X_needs_shift] -= X_total_minima[X_needs_shift, np.newaxis]

    X_total_maxima = X_total.max(axis=-1)
    X_needs_scaling = X_total_maxima > 1
    X_total[X_needs_scaling] /= X_total_maxima[X_needs_scaling, np.newaxis]
    return np.concatenate([X, X_total])
