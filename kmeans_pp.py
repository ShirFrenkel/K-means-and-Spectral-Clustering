# final version!
import mykmeanssp as ckm
import argparse
import pandas as pd
import numpy as np


def read_inputs():
    parser = argparse.ArgumentParser()

    parser.add_argument("K", help="the number of clusters required", type=int)
    parser.add_argument("N", help="the number of observations in the file", type=int)
    parser.add_argument("d", help="the dimension of each observation and initial centroids", type=int)
    parser.add_argument("MAX_ITER", help="the maximum number of iterations of the K-means algorithm", type=int)
    parser.add_argument("filename", help="the observation file path", type=str)

    args = parser.parse_args()
    K = args.K
    N = args.N
    d = args.d
    MAX_ITER = args.MAX_ITER
    file_path = args.filename

    # check inputs
    if len(vars(args)) != 5:
        print("number of arguments is incorrect (needs to be 5)")
        exit(1)
    if K >= N:
        print("K cannot be greater or equal to N")
        exit(1)
    if d < 1 or MAX_ITER < 1 or K < 1:
        print("one of the arguments given is not positive, shame on you")
        exit(1)

    # read file
    obs_df = pd.read_csv(file_path, header=None, dtype=np.float64)

    if N != obs_df.shape[0]:
        print("Number of formal and actual observations does not match!")
        exit(1)

    if d != obs_df.shape[1]:
        print("Number of formal and actual dimensions does not match!")
        exit(1)

    return K, N, d, MAX_ITER, obs_df


def calc_D(mu, x):
    """
    :param mu: np array, dim0- j dim 1- d
    :param x:  np array, dim0- N, dim1-d

    :return: D: np array, dim0- N, Di as described in the assignment HW2 pdf
    in a pythonic way: D = [min[(sum((x[i]-mu[z])**2) for z in range(j))] for i in range(N)]
    """

    y = (x - mu[:, np.newaxis]) ** 2  # y[z,i,:] = (x[i] - mu[z])**2
    distances = y.sum(axis=2)  # distances[z,i] = sum((x[i] - mu[z])**2)
    return np.min(distances, axis=0)  # D[i] = min[(sum((x[i]-mu[z])**2) for z in range(j))]


def k_means_pp(points, N, k, d):
    """
    param points: np array with dim0- N, dim1-d
    points[i,:] is d coordinates of observation i.
    :param K: amount of clusters to group the observations into
    :returns initial centroids for the k-mean algorithm & the indices of the observations chosen as initial centroids
    """
    np.random.seed(0)  # Seed randomness

    init_cent = np.zeros(shape=(k, d))  # initial centroids, dim0- k dim 1- d
    init_indices = []

    i = np.random.choice(N, 1)
    init_cent[0, :] = points[i, :]  # init mu 1
    init_indices.append(int(i[0]))

    for j in range(1, k):
        D = calc_D(init_cent[0:j, :], points)  # init_cent[0:j, :] - use only initiated centroids
        P = D / (D.sum())  # p[i] = probability of selecting x[i] as mu[j]
        i = np.random.choice(N, 1, p=P.tolist())
        init_cent[j, :] = points[i, :]  # init mu j: mu[j]= x[i]
        init_indices.append(int(i[0]))

    return init_cent.tolist(), init_indices


def kmeans_pp(K, N, d, MAX_ITER, obs):
    #K, N, d, MAX_ITER, obs_df = read_inputs() # DEL
    check_inputs()  # TODO
    #TODO: obs is not df now

    init_cent, init_indices = k_means_pp(obs_df.to_numpy(), N, K, d)
    print(','.join(str(x) for x in init_indices))

    obs_lst = obs_df.values.tolist()  # for passing to C implementation

    ckm.api_func(init_cent, obs_lst, K, N, d, MAX_ITER)




