import numpy as np

# TODO The Eigengap Heuristic
# TODO Algorithm 3 The Normalized Spectral Clustering Algorithm



"""
NOTES:
* modified_gram_schmidt- maybe working with transposed matrix is cheaper? is a[i] cheaper than a[:,i]?
* what to do if there is an error? just exit?
"""

# if you want to calc ||a-b||, just call calc_l2_norm(a-b)  # DEL
def calc_l2_norm(a):
    return np.sqrt(np.sum(a ** 2))


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
        R[i, i] = calc_l2_norm(U[:, i])
        if R[i, i] == 0:
            print("division by zero is undefined, exiting program")  # should not get here
            exit(1)
        Q[:, i] = U[:, i] / R[i, i]
        for j in range(i, n):
            R[i, j] = np.dot(Q[:, i].T, U[:, j])
            U[:, j] = U[:, j] - R[i, j] * Q[:, i]
    return Q, R

# NEW-----------------------------------------------

#A = np.array([[j for j in range(i,i+4)]for i in range(0,16,4)], np.float64)
#Q, R=modified_gram_schmidt(A)
# print(Q)
# print(R)
# print(np.dot(Q,R))


#DEL
def calc_weight():  # TODO
    pass
def normalized_graph_laplacian():  # TODO
    pass
def QR():  # TODO
    pass

def eigengap(eigenvalues):
    """
    :param eigenvalues: ndarray of eigenvalues in order
    :return: eigengap measure
    """

def normalized_spectral_clustering(points):
    """
    :param points: ndarry, shape(points) = (n,d) (n points with d dimensions)
    :return:
    """
    W = calc_weight(points)  # TODO
    Lnorm = normalized_graph_laplacian()  # TODO
    QR()  # TODO




