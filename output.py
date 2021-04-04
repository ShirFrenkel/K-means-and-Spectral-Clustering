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


def write_clusters_file(spectral_tags, kmeans_tags, k):
    """
        :param spectral_tags: list of cluster classification by the Normalized Spectral Clustering algorithm.
        :param kmeans_tags: list of cluster classification by the kmeans++ algorithm.
        :param k: the number of clusters to which we classified our date.
        @post: writing to clusters.txt in the format mentioned in the assigment.
    """
    spectral_for_print = convert_cluster_oriented(spectral_tags, k)
    kmeans_for_print = convert_cluster_oriented(kmeans_tags, k)
    string_to_print = str(k)
    for cluster in spectral_for_print:
        string_to_print += ('\n' + cluster)
    for cluster in kmeans_for_print:
        string_to_print += ('\n' + cluster)
    f = open("clusters.txt", "w")
    f.write(string_to_print)
    f.close()


def convert_cluster_oriented(cluster_tags, k):
    """
        :param cluster_tags: an array of n ints where cluster_tags[i] is the cluster number point i belongs to.
        :param k: the number of clusters.
        :return: list of strings in which converted[i] represents the points belonging to cluster i separated by ','.
    """
    converted = []
    for i in range(k):
        converted.append([])
    for i in range(len(cluster_tags)):
        converted[cluster_tags[i]].append(i)
    converted = [','.join([str(num) for num in cluster]) for cluster in converted]
    return converted


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
