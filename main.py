from sklearn.datasets import make_blobs
import random
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm


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

#DEL FUNC
# def generate_points(is_random, n=None, k=None):
#     """
#     :param n: amount of points
#     :param k: amount of centers
#     :param is_random: when is_random=True, n, k will be decided randomly with
#     the max capacity bound, the inputs n, k in that case are not used
#     :return: points, cluster_labels (for each point)
#     """
#     dim = random.randint(2, 3)  # dimensions for each point
#     if is_random:
#         n = random.randint(N_MAX_CAPACITY[dim]//2, N_MAX_CAPACITY[dim])
#         k = random.randint(K_MAX_CAPACITY[dim]//2, K_MAX_CAPACITY[dim])
#     points, cluster_labels = sklearn.datasets.make_blobs(n_samples=n, n_features=dim, centers=k)
#     return points, cluster_labels

def write_data_file(points, cluster_labels):
    print(points)
    print(cluster_labels)

    data = np.concatenate((points, cluster_labels[:, np.newaxis]), axis=1)  # axis = 1 means column-wise
    print(data)
    # write data to text file
    #np.savetxt(DATA_FILE_NAME, my_array, fmt="%4d", delimiter=",", newline="\n")  # TODO

def visualize(points, spec_cluster_map, kmeans_cluster_map ):
    k = 10
    colors = [cm.rainbow(i) for i in np.linspace(0, 1, k)]
    # print(colors[0])

    # x = np.linspace(0, 10, 30)
    # y1 = np.sin(x)
    # y2 = np.cos(x)
    #
    # plt.plot(x, y1, 'o', color='black')
    # plt.plot(x, y2, 'o', color=colors[0])
    #
    # plt.show()

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    zdata = 15 * np.random.random(100)
    xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
    ydata = np.cos(zdata) + 0.1 * np.randomd.randn(100)
    ax.scatter3D(xdata, ydata, zdata, c='red')

    plt.show()
visualize(None,None,None)


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
    # write to clusters.txt

    #run K++
    #write to clusters.txt

    #3d visio

    #the end?



#main(True)  # DEL




