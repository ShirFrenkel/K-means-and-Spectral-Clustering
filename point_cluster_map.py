class point_cluster_map:
    def __init__(self, name, map):
        """
        :param name: the name of the clustering algorithm that created the mapping
        :param map: map[i] is the index of the cluster that point i is belong to
        """
        self.name = name
        self.map = map