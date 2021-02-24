#define PY_SSIZE_T_CLEAN
#include <Python.h>
/* written by Tom Rutenberg*/

static void pList_to_cArray(PyObject*, Py_ssize_t, double**);
static void print_output(double **, int , int);
static void adding (double *c, double *, int);
static void calculate_updated_centroids(double **, double **, int *, int, int, int);
static double  distance(int, double *, double *);
static void devide_to_clusters(double **, double **, int *, int, int, int);
static int compare_centroids(double **, double **, int, int);


static void print_output(double **centroids, int num_clusters, int dimensions){
    int i;
    int j;
    printf("%f", centroids[0][0]);
    for (j = 1 ; j < dimensions ; j++)
            printf(",%f", centroids[0][j]);

    for (i = 1 ; i < num_clusters ; i++){
        printf("\n%f", centroids[i][0]);
        for (j = 1 ; j < dimensions ; j++)
            printf(",%f", centroids[i][j]);
         }

}

static void adding (double *centroid, double *observation, int dimension) {
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
    double result = 0;
    int i;
    for (i = 0 ; i < dimensions ; i++){
        result += (observation[i] - centroid[i])*(observation[i] - centroid[i]);
    }
    return result;
}

static void devide_to_clusters(double **observations, double **centroids, int *cluster_tags, int dimensions, int num_observations, int num_clusters){
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
    int i;
    int j;
    for (i = 0 ; i < num_clusters ; i++){
        for (j = 0 ; j < dimensions ; j++){
            if (centroids[i][j] != previous_centroids[i][j])
            return 0;
        }
    }
    return 1;
}

static void pList_to_cArray(PyObject* list, Py_ssize_t list_size, double** cArray){
    Py_ssize_t i, j, vector_size;
    PyObject *vector;
    PyObject *item;

    for (i = 0 ; i < list_size ; i++) {
        vector = PyList_GetItem(list, i);
        if (!PyList_Check(vector)){
           puts("vector is not a list");
           continue;
        }

        vector_size = PyList_Size(vector);
        cArray[i] = malloc(sizeof(double) * vector_size);
        assert(cArray[i] != NULL && "Problem in pList_to_cArray()");

        for (j =0 ; j < vector_size ; j++){
            item = PyList_GetItem(vector, j);
            if (!PyFloat_Check(item)){
                puts("Item not a Float");
                continue;
            }

            cArray[i][j] = PyFloat_AsDouble(item);
            if (cArray[i][j]  == -1 && PyErr_Occurred()){
                puts("Something bad ...");
                free(cArray[i]);
                return;
            }
        }
    }
    return;
}

static PyObject* api_func(PyObject *self, PyObject *args){
    PyObject *_pCentroids, *_pObservations;
    Py_ssize_t n;
    int  num_clusters, num_observations, dimensions, max_iter, i;
    int iteration = 0;
    double** centroids;
    double** observations;
    double** previous_centroids;
    double** tmp_centroids;
    int *cluster_tags;

    if(!PyArg_ParseTuple(args, "OOiiii", &_pCentroids, &_pObservations, &num_clusters, &num_observations, &dimensions, &max_iter)) {
        puts("argument parsing in api_func went bad");
        return NULL;
    }
    /*printf("k = %d, n = %d, d = %d, iter = %d\n", num_clusters, num_observations, dimensions, max_iter);*/

    centroids = (double **) malloc(num_clusters * sizeof(double *));
    if(centroids == NULL){
        puts("Problem allocating centroids");
        assert(centroids != NULL);
    }


    observations = (double **) malloc(num_observations * sizeof(double *));
    if(observations == NULL){
        puts("Problem allocating observations");
        assert(observations != NULL);
     }

    /* Might convert to parse_data() method*/
    if (!PyList_Check(_pCentroids)) {
        puts("Initial Centroids is not a list");
        return NULL;
    }
    if (!PyList_Check(_pObservations)) {
        puts("Observations is not a list");
        return NULL;
    }

    n = PyList_Size(_pCentroids);
    if (n != num_clusters){
        puts("Number of formal and actual clusters does not match!");
        return NULL;
    }
    pList_to_cArray(_pCentroids, n, centroids);



    n = PyList_Size(_pObservations);
    if (n != num_observations){
        puts("Number of formal and actual observations does not match!");
        return NULL;
    }
    pList_to_cArray(_pObservations, n, observations);

    previous_centroids = (double **) malloc(num_clusters * sizeof(double *));
    if (previous_centroids == NULL){
        puts("problem allocating previous centroids");
        assert(previous_centroids!=NULL);
    }

    for (i = 0 ; i < num_clusters ; i++){
         previous_centroids[i] = calloc(dimensions, sizeof(double));
         if (previous_centroids[i] == NULL){
            puts("problem allocating previous centroids[i]");
            assert(previous_centroids[i] != NULL);
         }
    }

    cluster_tags = calloc(num_observations, sizeof(int));
    if (cluster_tags == NULL){
        puts("problem allocating copy of cluster_tags");
        assert(cluster_tags != NULL);
    }


    while (iteration < max_iter){
        devide_to_clusters(observations, centroids, cluster_tags, dimensions, num_observations, num_clusters);
        tmp_centroids = previous_centroids;
        previous_centroids = centroids;
        centroids = tmp_centroids;
        calculate_updated_centroids(observations, centroids, cluster_tags, num_clusters, dimensions, num_observations);
        if (compare_centroids(centroids, previous_centroids, num_clusters, dimensions))
            break;
        iteration++;
    }

    print_output(centroids, num_clusters, dimensions);

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

    Py_RETURN_NONE;
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


