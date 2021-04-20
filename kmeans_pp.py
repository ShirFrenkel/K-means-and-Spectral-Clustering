import mykmeanssp as ckm
import argparse
import pandas as pd
import numpy as np


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


def kmeans_pp_main(K, MAX_ITER, obs):
    """
    :param K: the number of clusters required, 0 < K < N (N defined later)
    :param MAX_ITER: the maximum number of iterations of the K-means algorithm, 0 < MAX_ITER
    :param obs: the observations (points) to be clustered, numpy matrix of shape (N,d)
    N = the number of observations, d = the dimension of each observation
    :return: point_cluster_map !!! or None if error occurred !!!
    point_cluster_map[i] = the index of the cluster that point i is belong to (count starts from 0)"""

    N = obs.shape[0]
    d = obs.shape[1]
    init_cent, init_indices = k_means_pp(obs, N, K, d)

    return ckm.api_func(init_cent, obs.tolist(), K, N, d, MAX_ITER)

