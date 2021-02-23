import sklearn.datasets
import random

# maximum capacities
N_MAX_CAPACITY = {2: 100, 3: 50}  # TODO
K_MAX_CAPACITY = {2: 10, 3: 5}

"""
NOTES:
- is there a specific random module we should use? 
- do we care if int round is down and not up?
"""


def generate_points(is_random, n=None, k=None):
    """
    :param n: amount of points
    :param k: amount of centers
    :param is_random: when is_random=True, n, k will be decided randomly with
    the max capacity bound, the inputs n, k in that case are not used
    :return: points, cluster_labels (for each point)
    """
    dim = random.randint(2, 3)  # dimensions for each point
    if is_random:
        n = random.randint(N_MAX_CAPACITY[dim]//2, N_MAX_CAPACITY[dim])
        k = random.randint(K_MAX_CAPACITY[dim]//2, K_MAX_CAPACITY[dim])
    points, cluster_labels = sklearn.datasets.make_blobs(n_samples=n, n_features=dim, centers=k)
    return points, cluster_labels

if __name__ == "__main__":
    for i in range(2, 4):
        print(f'maximum capacity for {i}-dimensional data points:\n'
              f'\tnumber of centers (K) = {K_MAX_CAPACITY[i]}\n'
              f'\tnumber of data points (N) = {N_MAX_CAPACITY[i]}')
    #main() # TODO

    #generate_points(False,n=70,k=3)  # DEL





    """Testing commit and push"""