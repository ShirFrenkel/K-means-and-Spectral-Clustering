import numpy as np
from kmeans_pp import kmeans_pp

MAX_ITER = 300

# TODO The Eigengap Heuristic
# TODO Algorithm 3 The Normalized Spectral Clustering Algorithm



"""
NOTES:
* use #DEL for temp things
* use #TODO for future tasks

for TOM:
* for computing l2-norm, can use np.linalg.norm(vector, ord=2), no need to use my implementation (gonna delete later)

for Shir:
* modified_gram_schmidt- maybe working with transposed matrix is cheaper? is a[i] cheaper than a[:,i]?
* what to do if there is an error? just exit?
"""

# #DEL
# # if you want to calc ||a-b||, just call calc_l2_norm(a-b)  # DEL
# def calc_l2_norm(a):
#     return np.sqrt(np.sum(a ** 2))

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
    :param A: (ndarray) square matrix
    :return: Q = orthogonal matrix, R = upper triangular matrix (ndarrays)
    shape(A) = shape(Q) = shape(R) =(n,n)
    """
    U = A
    n = A.shape[0]
    Q = np.zeros((n,n), dtype=np.float64)
    R = np.zeros((n,n), dtype=np.float64)

    for i in range(n):
        R[i, i] = np.linalg.norm(U[:, i], ord=2)
        if R[i, i] == 0:
            print("division by zero is undefined, exiting program")  # should not get here
            exit(1)
        Q[:, i] = U[:, i] / R[i, i]
        for j in range(i, n):
            R[i, j] = np.dot(Q[:, i].T, U[:, j])
            U[:, j] = U[:, j] - R[i, j] * Q[:, i]
    return Q, R

# NEW-----------------------------------------------

# #test gram_shmidt
# A = np.array([[j for j in range(i,i+4)]for i in range(0,16,4)], np.float64)
# Q, R=modified_gram_schmidt(A)
# print(Q)
# print(R)
# print(np.dot(Q,R))

# TOM'S FUNCTIONS


def calc_weight(points):
    """
        :param points: nxd matrix, each row is a point
        :return: w = The Weighted Adjacency Matrix. n dimensions square, symmetric and non-negative matrix
        """
    n = points.shape[0]
    w = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):  # W[i][i] is already set to zero as needed
            exponent = np.linalg.norm(points[i] - points[j]) ** 2 / (-2)
            w[i][j] = np.math.exp(exponent)
            w[j][i] = w[i][j]
    return w


def normalized_graph_laplacian(W, D):
    l_norm = np.identity(W.shape[0])
    l_norm -= D @ W @ D
    return l_norm


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

def normalized_spectral_clustering(points):
    """
    :param points: ndarry, shape(points) = (n,d) (n points with d dimensions)
    :return:
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

    k = eigengap(eigenvalues)  # number of clusters for the clustering
    U = eignvectors[:, 0:k]  # U.shape = (n,k)
    U_norm = normalize_rows(U)  # U_norm := U with normalized rows
    # Treating each row of U_norm as a point in Rk, cluster them into k clusters via the K-means algorithm:
    kmeans_pp(k, U_norm.shape[0], k, MAX_ITER, U_norm)  # arguments meaning: K, N, d, MAX_ITER, obs
    # read the output file... #TODO





