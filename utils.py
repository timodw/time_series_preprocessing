import numpy as np
from sklearn.neighbors import NearestNeighbors

def find_k_nearest_neighbors(X, k):
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors.fit(X)
    distances, indices = neighbors.kneighbors(X)
    return distances, indices

def generate_synthetic_samples(X, k_neighbors, n_samples):
    n_features = X.shape[1]
    synthetic_samples = np.zeros((n_samples, n_features))
    for i in range(n_samples):
        sample_idx = np.random.randint(0, X.shape[0])
        neighbor_idx = np.random.choice(k_neighbors[sample_idx])
        diff = X[neighbor_idx] - X[sample_idx]
        synthetic_samples[i] = X[sample_idx] + np.random.rand() * diff
    return synthetic_samples
