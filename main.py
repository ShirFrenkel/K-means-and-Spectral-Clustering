from sklearn.datasets import make_blobs
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

MAX_ITER = 300

DATA_FILE_NAME = "data.txt"
CLUSTERS_FILE_NAME = "clusters.txt"

# maximum capacities
N_MAX_CAPACITY = {2: 100, 3: 50}  # TODO
K_MAX_CAPACITY = {2: 10, 3: 5}

"""
NOTES:
- is there a specific random module we should use? 
- do we care if int round is down and not up?
- when is_random, is there an option that k > n? (if so, we have a problem...)
- when writing to data.txt, do we need to write the whole num? or round it to some point ?
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


def write_data_file(points, cluster_labels):
    #print(points)
    #print(cluster_labels)
    data = np.concatenate((points, cluster_labels[:, np.newaxis]), axis=1)  # axis = 1 means column-wise
    #print(data)
    # write data to text file
    np.savetxt(DATA_FILE_NAME, data, delimiter=",", newline="\n")
    #TODO not sure what is the accuracy level we need for points & need to fix integer apears as float


def jaccard_measure():
    return 0  # TODO


def visualize(points, k, point_cluster_maps):
    """
    :param points: ndarry, shape(points) = (n,d) (n points with d dimensions)
    :param k: amount of clusters
    :param point_cluster_maps:
    point_cluster_maps[0] - Normalized Spectral Clustering points map
    point_cluster_maps[1] - K-means Clustering points map
    (lst is points map if lst[i] is the index of the cluster that point i is belong to)
    @post: creates clusters.pdf
    """
    # TODO ? change to T order
    dim = points.shape[1]
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2] if dim == 3 else None
    projection = '3d' if dim == 3 else None

    fig, ax = plt.subplots(1, 2, subplot_kw=dict(projection=projection))  # 2 subplots horizontally

    colors = [cm.rainbow(i) for i in np.linspace(0, 1, k)]
    titles = ["Normalized Spectral Clustering", "K-means"]

    for i in [0, 1]:
        color = [colors[cluster] for cluster in point_cluster_maps[i]]  # color for each point by it's cluster
        ax[i].scatter(x, y, z, c=color)
        ax[i].set_title(titles[i])

    plt.figtext(0.3, 0, "insert an informative desc as requested!!!\n\n****")  # TODO
    plt.show()
    #plt.savefig("clusters.pdf")


# test visualize
A = np.array([[j for j in range(i,i+3)]for i in range(0,15,3)], np.float64)
visualize(A,3,[[0,1,0,1,0],[2,2,2,1,1]])

B = np.array([[j for j in range(i,i+2)]for i in range(0,10,2)], np.float64)
visualize(B,3,[[0,1,0,1,0],[2,2,2,1,1]])


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
              f'\tnumber of centers (K) = {K_MAX_CAPACITY[i]}\n'
              f'\tnumber of data points (N) = {N_MAX_CAPACITY[i]}')

    dim = random.randint(2, 3)  # dimensions for each point
    if is_random:
        n = random.randint(N_MAX_CAPACITY[dim]//2, N_MAX_CAPACITY[dim])
        k = random.randint(K_MAX_CAPACITY[dim]//2, K_MAX_CAPACITY[dim])

    else:
        check_input(n,k)

    points, cluster_labels = make_blobs(n_samples=n, n_features=dim, centers=k)  # generate points

    write_data_file(points, cluster_labels)
    #TODO
    #run spectral
    # write to clusters.txt (or from C?)

    #run K++
    #write to clusters.txt (or from C?)

    #3d visio

    #the end?



main(True)  # DEL




