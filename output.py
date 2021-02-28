import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import config
from point_cluster_map import point_cluster_map

def write_data_file(points, cluster_labels):
    #print(points)
    #print(cluster_labels)
    data = np.concatenate((points, cluster_labels[:, np.newaxis]), axis=1)  # axis = 1 means column-wise
    #print(data)
    # write data to text file
    np.savetxt(config.DATA_FILE_NAME, data, delimiter=",", newline="\n")
    #TODO not sure what is the accuracy level we need for points & need to fix integer apears as float


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