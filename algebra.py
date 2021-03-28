import numpy as np
from kmeans_pp import kmeans_pp_main
import config

"""
NOTES:
* add epsilon to prev pro (hw2)
* use #DEL for temp things
* use #TODO for future tasks

for TOM:
* for computing l2-norm, can use np.linalg.norm(vector, ord=2), no need to use my implementation (gonna delete later)

for Shir:

"""


#for computing l2-norm, can use np.linalg.norm(vector, ord=2)  # DEL

# a = np.arange(16).reshape(4,4)


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
    * function uses fortran order (order='F') for efficient columns operations
    """
    n = A.shape[0]
    U = np.copy(A, order='F')
    Q = np.zeros((n, n), dtype=np.float64, order='F')
    R = np.zeros((n, n), dtype=np.float64, order='F')

    for i in range(n):
        R[i, i] = np.linalg.norm(U[:, i], ord=2)
        if R[i, i] == 0:
            print("division by zero is undefined, exiting program")  # should not get here
            exit(1)
        Q[:, i] = U[:, i] / R[i, i]
        R[i, i+1:] = Q[:, i] @ U[:, i+1:]
        U[:, i+1:] = U[:, i+1:] - (R[np.newaxis, i,  i+1:] * Q[:, i, np.newaxis])

    return Q, R


# def modified_gram_schmidt_for(A):  # DEL
#     """
#     :param A: (ndarray) square matrix with dtype = np.float64
#     :return: Q = orthogonal matrix, R = upper triangular matrix (ndarrays)
#     shape(A) = shape(Q) = shape(R) =(n,n)
#     * function uses fortran order (order='F') for efficient columns operations
#     """
#     n = A.shape[0]
#     U = np.copy(A, order='F')
#     Q = np.zeros((n, n), dtype=np.float64, order='F')
#     R = np.zeros((n, n), dtype=np.float64, order='F')
#
#     for i in range(n):
#         R[i, i] = np.linalg.norm(U[:, i], ord=2)
#         if R[i, i] == 0:
#             print("division by zero is undefined, exiting program")  # should not get here
#             exit(1)
#         Q[:, i] = U[:, i] / R[i, i]
#         for j in range(i+1, n):  # TODO- can do this without loops!
#             R[i, j] = Q[:, i] @ U[:, j]
#             U[:, j] = U[:, j] - R[i, j] * Q[:, i]
#
#         # U[:, i:n] = U[:, i:n] - (R[i,  i:n])[np.newaxis, :] * (Q[:, i])[:, np.newaxis]
#
#     return Q, R


# #test gram_shmidt
# A = np.array([[j for j in range(i,i+4)]for i in range(0,16,4)], np.float64)
# A = np.array([[1,2,3],[10,20,30],[11,5,7]], np.float64)
# A = np.array(100 * np.random.ranf(36), dtype=np.float64).reshape((6, 6))
# print(A)
# Q, R=modified_gram_schmidt(A)
# print(Q)
# print(R)
# # print(np.isfortran(Q))
# # print(np.isfortran(R))
# print((Q@R))
#
# q,r = np.linalg.qr(A, mode='reduced')
# #q,r = modified_gram_schmidt_for(A)
# # print(q)
# # print(r)
# print(np.absolute(R)-np.absolute(r))
# print(np.absolute(Q)-np.absolute(q))

# SQR- I this you can do this functions without loops, using np coding (working with matrices),
# you can read kmeans_pp.calc_D that I wrote, maybe it can help you get this idea
def calc_weight(points):
    """
        :param points: nxd matrix, each row is a point
        :return: w = The Weighted Adjacency Matrix. n dimensions square, symmetric and non-negative matrix
        """
    w = ((points - points[:, np.newaxis]) ** 2).sum(axis=2)
    return np.exp(w / -2) - np.identity(w.shape[0])

# SQR- np.identity ruins the idea of 1 dim matrix that I've created,
# maybe check if will be more efficient if you do scalar * rows and scalar* columns as we talked before
def normalized_graph_laplacian(W, D):
    l_norm = np.identity(W.shape[0])
    l_norm -= D @ W @ D
    return l_norm

# SQR- I dont think we use A afterwards
def QR(A):
    A1 = A.copy()  # creating a copy so we don't change A, if A is not used afterwards we can overwrite
    n = A.shape[0]
    Q1 = np.identity(n)
    for i in range(n):
        Q, R = modified_gram_schmidt(A1)
        A1 = R @ Q
        if converged(Q1, Q1 @ Q):
            return A1, Q1
        Q1 = Q1 @ Q
    return A1, Q1   # A1 = eigenvalues, Q1 = eigenvectors


def converged(a, b):
    dif = np.absolute(a) - np.absolute(b)
    dif = np.absolute(dif) <= 0.0001
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
    return int(np.argmax(delta))

#DEL
# a = np.array([2,6.4,10.8,11,11,11,11,2,11,11,11], np.float64)
# indices = np.argsort(a)  # indices that would sort hte matrices by increasing eigenvalues
# a = a[indices]
# print(a)
# print(eigengap(a))


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
    :return: point_cluster_map, k !!! or None, k if error occurred !!!
    point_cluster_map[i] = the index of the cluster that point i is belong to (count starts from 0)
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
    point_cluster_map = kmeans_pp_main(k, config.MAX_ITER, U_norm)  # arguments meaning: K, MAX_ITER, obs matrix
    # point_cluster_map[i] = the index of the cluster that U_norm's row i is belong to = ...
    # ... = the index of the cluster that point i is belong to
    return point_cluster_map, k








