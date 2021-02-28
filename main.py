from sklearn.datasets import make_blobs
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import config

"""
NOTES:
- is there a specific random module we should use? 
- do we care if int round is down and not up?
- when is_random, is there an option that k > n? (if so, we have a problem...)
- when writing to data.txt, do we need to write the whole num? or round it to some point ?
"""


class point_cluster_map:
    def __init__(self, name, map):
        """
        :param name: the name of the clustering algorithm that created the mapping
        :param map: map[i] is the index of the cluster that point i is belong to
        """
        self.name = name
        self.map = map


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
    np.savetxt(config.DATA_FILE_NAME, data, delimiter=",", newline="\n")
    #TODO not sure what is the accuracy level we need for points & need to fix integer apears as float


def jaccard_measure():
    return 0  # TODO


def visualize(points, k_original, k_algo, maps_lst):
    """
    :param points: ndarry, shape(points) = (n,d) (n points with d dimensions)
    :param k_original: amount of centers the points were generated from
    :param k_algo: k that was used for both algorithms
    :param maps_lst: lst of point_cluster_map objects matching the input's points
    @post: creates clusters.pdf with axis for each map in maps_lst ordered horizontally by the maps_lst order
    """
    # TODO ? change to T order
    dim = points.shape[1]
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2] if dim == 3 else None
    projection = '3d' if dim == 3 else None

    fig, ax = plt.subplots(1, len(maps_lst), subplot_kw=dict(projection=projection))  # subplots horizontally

    colors = [cm.rainbow(i) for i in np.linspace(0, 1, k_algo)]

    for i in range(len(maps_lst)):
        color = [colors[cluster] for cluster in maps_lst[i].map]  # color for each point by it's cluster
        ax[i].scatter(x, y, z, c=color)
        ax[i].set_title(maps_lst[i].name)

    plt.figtext(0.3, 0, "insert an informative desc as requested!!!\n\n****")  # TODO, maybe use subtitle instead of this shit
    #plt.show()
    plt.savefig("clusters.pdf")


# test visualize
spec = point_cluster_map("Normalized Spectral Clustering", [0,1,0,1,0])
kmeans = point_cluster_map("K-means", [2,2,2,1,1])
shir_algo = point_cluster_map("another algo", [0,0,0,0,0])
lst_map = [spec, kmeans, shir_algo]

A = np.array([[j for j in range(i,i+3)]for i in range(0,15,3)], np.float64)
visualize(A,3,3,lst_map)

B = np.array([[j for j in range(i,i+2)]for i in range(0,10,2)], np.float64)
visualize(B,3,3,lst_map)


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








