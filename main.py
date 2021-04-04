from sklearn.datasets import make_blobs
import random
import numpy as np
import config
from output import write_data_file, visualize, write_clusters_file
from point_cluster_map import point_cluster_map  # need this for later
from algebra import normalized_spectral_clustering
from kmeans_pp import kmeans_pp_main

"""
NOTES:
- is there a specific random module we should use? 
- do we care if int round is down and not up?
- when is_random, is there an option that k > n? (if so, we have a problem...)
- when writing to data.txt, do we need to write the whole num? or round it to some point ?
- check error in C case 
"""


def check_input(n, k):
    if n is None or k is None:
        print("K or N is not supplied")
        exit(1)
    if k >= n:
        print("K cannot be greater or equal to N")
        exit(1)
    if k < 1 or n < 1:
        print("one of the arguments given is not positive, shame on you")
        exit(1)


def jaccard_measure():  # maybe should be in different module (in output.py?)
    return 0  # TODO


def main(is_random, n=None, k=None):
    """
    :param n: amount of points
    :param k: amount of centers
    :param is_random: when is_random=True, n, k will be decided randomly with
    the max capacity bound, the inputs n, k in that case are not used
    :return: #TODO
    """
    # print max capacities
    for i in [2,3]:
        print(f'maximum capacity for {i}-dimensional data points:\n'
              f'\tnumber of centers (K) = {config.K_MAX_CAPACITY[i]}\n'
              f'\tnumber of data points (N) = {config.N_MAX_CAPACITY[i]}')

    dim = random.randint(2, 3)  # dimensions for each point
    if is_random:
        n = random.randint(config.N_MAX_CAPACITY[dim]//2, config.N_MAX_CAPACITY[dim])
        k = random.randint(config.K_MAX_CAPACITY[dim]//2, config.K_MAX_CAPACITY[dim])

    else:
        check_input(n, k)

    points, cluster_labels = make_blobs(n_samples=n, n_features=dim, centers=k)  # generate points

    write_data_file(points, cluster_labels)

    spectral_cluster_tags, k = normalized_spectral_clustering(points, is_random, k)
    if spectral_cluster_tags is None:
        print("An error occurred during normalized spectrael clustring, shutting down")
        exit(1)

    kmeans_cluster_tags = kmeans_pp_main(k, config.MAX_ITER, points)
    if kmeans_cluster_tags is None:
        print("An error occurred during kmeans++, shutting down")
        exit(1)

    write_clusters_file(spectral_cluster_tags, kmeans_cluster_tags, k)

    # TODO

    #3d visio
    # visualize(...)

    #the end?



main(True)  # DEL








