import numpy as np
from kmeans_pp import kmeans_pp_main
import config

"""This module contains the functions which implement the Normalized Spectral Clustering algorithm.
    in order to use it you only need to invoke normalized_spectral_clustering() function
    with the following arguments: a data set of points, a is_random flag and a number of clusters k.
"""

def diagonal_degree_matrix(W):
    """
    W = The Weighted Adjacency Matrix
    :return: D, D[i] = sum(W[i])**(-1/2)
    """
    return np.power(np.sum(W, axis=1), -0.5)


def modified_gram_schmidt(A):
    """
    :param A: (ndarray) square matrix with dtype = np.float64
    :return: Q = orthogonal matrix, R = upper triangular matrix (ndarrays)
    shape(A) = shape(Q) = shape(R) =(n,n)
    * function changes A
    """
    n = A.shape[0]
    Q = np.zeros((n, n), dtype=np.float64)
    R = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        R[i, i] = np.linalg.norm(A[:, i], ord=2)
        if R[i, i] == 0:
            print("division by zero is undefined, exiting program")  # should not get here
            exit(1)
        Q[:, i] = A[:, i] / R[i, i]
        R[i, i+1:] = Q[:, i] @ A[:, i+1:]
        A[:, i+1:] = A[:, i+1:] - (R[i,  i+1:] * Q[:, i, np.newaxis])

    return Q, R


def calc_weight(points):
    """
        :param points: nxd matrix, each row is a point
        :return: w = The Weighted Adjacency Matrix. n dimensions square, symmetric and non-negative matrix
        """
    w = np.linalg.norm((points - points[:, np.newaxis]), ord=2, axis=2)
    return np.exp(w / -2) - np.identity(w.shape[0])


def normalized_graph_laplacian(W, D):
    """
        :param D: The Diagonal Degree Matrix raised to the power of -0.5 (represented as 1D vector!)
        :param W: The Weighted Adjacency Matrix
        :return: l_norm = The normalized graph Laplacian
    """
    l_norm = np.identity(W.shape[0]) - (D[np.newaxis, :] * W * D[:, np.newaxis])
    return l_norm


def QR(A):
    n = A.shape[0]
    Q1 = np.identity(n, dtype=np.float64)
    for i in range(n):
        Q, R = modified_gram_schmidt(A)
        A = R @ Q
        if converged(Q1, Q1 @ Q):
            return A, Q1
        Q1 = Q1 @ Q
    return A, Q1   # A = eigenvalues, Q1 = eigenvectors


def converged(a, b):
    dif = np.absolute(a) - np.absolute(b)
    dif = np.absolute(dif) <= config.EPSILON
    return np.alltrue(dif)


def eigengap(eigenvalues):
    """
    :param eigenvalues: ndarray of eigenvalues **sorted**
    :return: eigengap measure (int)
    In case of equality in the argmax of some eigengaps, use the lowest index
    """
    half_len = len(eigenvalues)//2
    delta = eigenvalues[1:half_len+1] - eigenvalues[:half_len]
    # no need for abs because eigenvalues are sorted by increasing value
    return int(np.argmax(delta)) + 1


def normalize_rows(M):
    """
    :param M: 2-dim ndarray
    :return: M normalized (by rows)
    """
    return M / np.linalg.norm(M, ord=2, axis=1, keepdims=True)

def normalized_spectral_clustering(points, is_random, k=None):
    """
    :param points: ndarry, shape(points) = (n,d) (n points with d dimensions)
    if is_random, k is computed by eigengap heuristic, else, k is supplied as input
    :return: mapping, k !!! or None, k if error occurred !!!
    mapping[i] = the index of the cluster that point i is belong to (count starts from 0)
    """
    W = calc_weight(points)
    Lnorm = normalized_graph_laplacian(W, diagonal_degree_matrix(W))
    eigenvalues, eignvectors = QR(Lnorm)
    # each eigenvalue- eigenvalues[j,j] corresponds to an eigenvector- eignvectors[:,j] (approximately)
    eigenvalues = eigenvalues.diagonal()

    # sort eigenvalues, eignvectors by increasing eigenvalues
    indices = np.argsort(eigenvalues)  # indices that would sort hte matrices by increasing eigenvalues
    eigenvalues = eigenvalues[indices]
    eignvectors = eignvectors[:, indices]

    if is_random:
        k = eigengap(eigenvalues)  # number of clusters for the clustering
    U = eignvectors[:, 0:k]  # U.shape = (n,k)
    U_norm = normalize_rows(U)  # U_norm := U with normalized rows
    # Treating each row of U_norm as a point in Rk, cluster them into k clusters via the K-means algorithm:
    mapping = kmeans_pp_main(k, config.MAX_ITER, U_norm)  # arguments meaning: K, MAX_ITER, obs matrix
    # mapping[i] = the index of the cluster that U_norm's row i is belong to = ...
    # ... = the index of the cluster that point i is belong to
    return mapping, k
