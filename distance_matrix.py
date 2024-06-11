from numba import njit
from multiprocessing import Pool
from scipy.spatial.distance import euclidean
import numpy as np
import ctypes
libdtw = ctypes.CDLL('./libdtw.so')

# Specify the argument and result types
libdtw.dtw.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
libdtw.dtw.restype = ctypes.c_double

# Python function to wrap the C function
def dtw_c(s1, s2):
    array_type1 = ctypes.c_double * len(s1)
    array_type2 = ctypes.c_double * len(s2)
    result = libdtw.dtw(len(s1), array_type1(*s1), len(s2), array_type2(*s2))
    return result


@njit
def dtw_njit(s1, s2):
    len_s1 = len(s1)
    len_s2 = len(s2)
    mat_d = np.full((len_s1 + 1, len_s2 + 1), np.inf, dtype=np.float64)
    mat_d[0, 0] = 0.0

    for i in range(1, len_s1 + 1):
        for j in range(1, len_s2 + 1):
            d = (s1[i - 1] - s2[j - 1])**2
            mat_d[i, j] = d + np.min(np.array([mat_d[i-1, j], mat_d[i, j-1], mat_d[i-1, j-1]]))
    
    return np.sqrt(mat_d[len_s1, len_s2])


def compute_row(args):
    i, X, n, metric = args
    if metric == 'dtw':
        distance_fn = dtw_c
    else:
        distance_fn = euclidean
    distances = np.zeros(n)
    for j in range(i, n):
        distances[j] = distance_fn(X[i], X[j])
    return i, distances


def parallel_distance_matrix(X, metric='dtw'):
    n = len(X)
    distance_matrix = np.zeros((n, n))
    
    args = [(i, X, n, metric) for i in range(n)]
    
    with Pool() as pool:
        results = pool.map(compute_row, args)
    
    for i, distances in results:
        distance_matrix[i, i:] = distances[i:]
        distance_matrix[:, i][i:] = distances[i:]

    return distance_matrix