#%%
#for algebra
import numpy as np
from kmeans_pp import kmeans_pp_main
import config

#for main
from sklearn.datasets import make_blobs
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import config

#%%
a = np.arange(16).reshape(4,4)
#%%
# test visualize
from main import point_cluster_map, visualize
spec = point_cluster_map("Normalized Spectral Clustering", [0,1,0,1,0])
kmeans = point_cluster_map("K-means", [2,2,2,1,1])
shir_algo = point_cluster_map("another algo", [0,0,0,0,0])
lst_map = [spec, kmeans, shir_algo]

A = np.array([[j for j in range(i,i+3)]for i in range(0,15,3)], np.float64)
visualize(A,3,3,lst_map)

B = np.array([[j for j in range(i,i+2)]for i in range(0,10,2)], np.float64)
visualize(B,3,3,lst_map)


#%%