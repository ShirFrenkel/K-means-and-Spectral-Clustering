#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define EPSILON 0.0001
/* This is the C extension for the K-means algorithm
    You invoke it from a python program (in this case kmeans_pp.py) by calling api_func.
    The returned value is a python list of ints, as many as the number of points in the dataset passed as an argument
    where the value in index i is the cluster to which point was classified by the algorithm.
 */

static void pList_to_cArray(PyObject*, int, double**);
static void adding (double *c, double *, int);
static void calculate_updated_centroids(double **, double **, int *, int, int, int);
static double distance(int, double *, double *);
static void divide_to_clusters(double **, double **, int *, int, int, int);
static int compare_centroids(double **, double **, int, int);
static int validate_input(PyObject*, PyObject*, int, int);
static int allocate_memory(double***, double***, double***, int**, int n, int k, int d);


static void adding (double *centroid, double *observation, int dimension) {
    /* a helper function used to calculate the centroid of a cluster */
    int i;
    for (i = 0 ; i < dimension ; i++)
        centroid[i] += observation[i];
}

static void calculate_updated_centroids(double **observations, double **centroids, int *cluster_tags, int num_of_clusters, int dimensions, int num_observations){
    int i;
    for (i = 0 ; i < num_of_clusters ; i++){ /* for each cluster */
        int j;
        int count = 0;
        for (j = 0 ; j < dimensions ; j++)   /* reset centroid */
            centroids[i][j] = 0;
        for  (j = 0 ; j < num_observations ; j++){ /* for each observation */
            if (cluster_tags[j] == i){                   /*if observations assigned to cluster */
                adding(centroids[i], observations[j], dimensions);
                count++;
            }
        }
        for (j = 0 ; j < dimensions ; j++){
            centroids[i][j] /= count;
        }
    }

}

static double distance(int dimensions, double *observation, double *centroid){
    /* computes squared distance (with l2 norm) */
    double result = 0;
    int i;
    for (i = 0 ; i < dimensions ; i++){
        result += (observation[i] - centroid[i])*(observation[i] - centroid[i]);
    }
    return result;
}

static void divide_to_clusters(double **observations, double **centroids, int *cluster_tags, int dimensions, int num_observations, int num_clusters){
    /* a function that assigns each observation to it's closest centroid */
    int i;
    for (i = 0 ; i < num_observations ; i++){ /* for each observation */
        double min_dist = distance(dimensions, observations[i], centroids[0]);
        int group = 0;
        int j;
        for (j = 1 ; j < num_clusters ; j++){ /* calculate distance from each centroid and return min */
            if (distance(dimensions, observations[i], centroids[j]) < min_dist){
                min_dist = distance(dimensions, observations[i], centroids[j]);
                group = j;
            }
        }
    cluster_tags[i] = group;
    }
}

static int compare_centroids(double **centroids, double **previous_centroids, int num_clusters, int dimensions){
    /* a function which determines whether the centroids have changed in the past iteration */
    int i;
    for (i = 0 ; i < num_clusters ; i++){
            /* distance() actually returns distance square so I rise epsilon to the power of two */
            if (distance(dimensions, centroids[i], previous_centroids[i]) > EPSILON*EPSILON)
                return 0;
    }
    return 1;
}

static void pList_to_cArray(PyObject* list, int list_size, double** cArray){
    /* a function which takes a python list of lists passed as an arguments and stores it in a 2D C array */
    int i;
    Py_ssize_t j, vector_size;
    PyObject *vector;
    PyObject *item;

    for (i = 0 ; i < list_size ; i++) {
        vector = PyList_GetItem(list, i);
        if (!PyList_Check(vector)){
           puts("vector is not a list");
           continue;
        }

        vector_size = PyList_Size(vector);

        for (j =0 ; j < vector_size ; j++){
            item = PyList_GetItem(vector, j);
            if (!PyFloat_Check(item)){
                puts("Item not a Float");
                continue;
            }

            cArray[i][j] = PyFloat_AsDouble(item);
            if (cArray[i][j]  == -1 && PyErr_Occurred()){
                puts("Something bad happened while converting python float to C double...");
                continue;
            }
        }
    }
    return;
}

static int validate_input(PyObject* _pCentroids, PyObject* _pObservations, int num_clusters, int num_observations){
    /* a function that verifies that the number of points and clusters passed matches the dataset passed */

    Py_ssize_t n;

    if (!PyList_Check(_pCentroids)) {
        puts("The argument passed as initial centroids is not a list as it should be");
        return 0;
    }

    n = PyList_Size(_pCentroids);
    if (n != num_clusters){
        puts("Number of formal and actual clusters does not match!");
        return 0;
    }

    if (!PyList_Check(_pObservations)) {
        puts("The argument passed as observations is not a list as it should be");
        return 0;
    }

    n = PyList_Size(_pObservations);
    if (n != num_observations){
        puts("Number of formal and actual observations does not match!");
        return 0;
    }
    return 1;
}

static int allocate_memory(double*** centroids, double*** observations, double*** previous_centroids, int** cluster_tags, int n, int k, int d){
    /* a function that allocated all memory needed for the algorithm, returns 1 on successful run, 0 otherwise */

    int i;
    *centroids = (double **) malloc(k * sizeof(double *));
    if(*centroids == NULL){
        puts("Problem allocating memory for centroids matrix");
        return 0;
    }

    for (i = 0 ; i < k ; i++){
         (*centroids)[i] = calloc(d, sizeof(double));
         if ((*centroids)[i] == NULL){
            puts("Problem allocating memory for one of the centroids");
            return 0;
         }
    }

    *observations = (double **) malloc(n * sizeof(double *));
    if(*observations == NULL){
        puts("Problem allocating memory for observations matrix");
        return 0;
     }
    for (i = 0 ; i < n ; i++){
         (*observations)[i] = calloc(d, sizeof(double));
         if ((*observations)[i] == NULL){
            puts("Problem allocating memory for one of the observations");
            return 0;
         }
    }

    *previous_centroids = (double **) malloc(k * sizeof(double *));
    if (*previous_centroids == NULL){
        puts("Problem allocating memory for previous centroids matrix");
        return 0;
    }

    for (i = 0 ; i < k ; i++){
         (*previous_centroids)[i] = calloc(d, sizeof(double));
         if ((*previous_centroids)[i] == NULL){
            puts("Problem allocating memory for a previous centroid");
            return 0;
         }
    }

    *cluster_tags = calloc(n, sizeof(int));
    if (*cluster_tags == NULL){
        puts("Problem allocating memory for cluster tags array");
        return 0;
    }
    return 1;

}

static PyObject* api_func(PyObject *self, PyObject *args){
    /* the "main" function of the C extension, takes the following arguments:
        list of centroids, list of points, number of clusters, number of points, the dimension and max number of iterations.
    */

    PyObject *_pCentroids, *_pObservations;
    int  num_clusters, num_observations, dimensions, max_iter, i;
    int iteration = 0;
    double** centroids;
    double** observations;
    double** previous_centroids;
    double** tmp_centroids;
    int *cluster_tags;

    if(!PyArg_ParseTuple(args, "OOiiii", &_pCentroids, &_pObservations, &num_clusters, &num_observations, &dimensions, &max_iter)) {
        puts("argument parsing in api_func went bad");
        Py_RETURN_NONE; /*return None to python so it will know something went bad */
    }

    if (validate_input(_pCentroids, _pObservations, num_clusters, num_observations) == 0){
        puts("Formal and actual arguments passed to C extension did not match");
        Py_RETURN_NONE;
    }

    if (allocate_memory(&centroids, &observations, &previous_centroids, &cluster_tags, num_observations, num_clusters, dimensions) == 0){
        puts("Problem allocating memory in C extension");
        Py_RETURN_NONE;
    }

    pList_to_cArray(_pCentroids, num_clusters, centroids);
    pList_to_cArray(_pObservations, num_observations, observations);

    while (iteration < max_iter){
        divide_to_clusters(observations, centroids, cluster_tags, dimensions, num_observations, num_clusters);
        tmp_centroids = previous_centroids;
        previous_centroids = centroids;
        centroids = tmp_centroids;
        calculate_updated_centroids(observations, centroids, cluster_tags, num_clusters, dimensions, num_observations);
        if (compare_centroids(centroids, previous_centroids, num_clusters, dimensions))
            break;
        iteration++;
    }

    PyObject* py_lst = PyList_New(num_observations); /* creating the PyObject which will be returned*/
    for (i = 0 ; i < num_observations ; i++) {
        PyObject* python_int = Py_BuildValue("i", cluster_tags[i]);
        PyList_SetItem(py_lst, i, python_int);
    }

    for (i = 0 ; i < num_clusters ; i++) {
        free(centroids[i]);
        free(previous_centroids[i]);
    }
    for (i = 0 ; i < num_observations ; i++)
         free(observations[i]);

    free(observations);
    free(centroids);
    free(previous_centroids);
    free(cluster_tags);

    return py_lst;
}

#define FUNC(_flag, _name, _docstring) { #_name, (PyCFunction)_name, _flag, PyDoc_STR(_docstring) }

static PyMethodDef _methods[] = {
    FUNC(METH_VARARGS, api_func, "the function that connects the python script with the C module"),
    {NULL, NULL, 0, NULL}   /* sentinel */
};

static struct PyModuleDef _moduledef = {
    PyModuleDef_HEAD_INIT,
    "mykmeanssp",
    NULL,
    -1,
    _methods
};

PyMODINIT_FUNC
PyInit_mykmeanssp(void)
{
    return PyModule_Create(&_moduledef);
}


