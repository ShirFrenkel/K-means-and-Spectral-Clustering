# K-means & Spectral Clustering 

## Description
### Essence
* Implementations of K-means and the Normalized Spectral Clustering algorithms from scratch.
* Comparison between the two algorithms by Jaccard measure and visualization.
* Written in Python and C using Numpy, Matplotlib and CAPI.
* The final project for Software Project course at TAU.

### Program Flow & Outputs
Generates data points, clusters the data using both algorithms and creates 3 output files:
* `data.txt` - the generated points and the corresponding center of each. 
* `clusters.txt` - contains the computed clusters from both algorithms. 
	- The first value will be the number of clusters k that was used. 
	- The next k lines will contain in each line the indices of points belonging to the same cluster 
	computed by the Normalized Spectral Clustering algorithm. 
	- The last k lines will have the same format, but this time of the resulting K-means clusters. 
	**Note:** When using the indices we refer to the first point in data.txt as having the index 0, the second as 1 and so on.
* `clusters.pdf` - a visualization of the clustering, results summary and score for each clustering by **Jaccard measure**.

  
## Demo
clusters.pdf visualization created by `$ python -m invoke run -k=5 -n=150 --no-Random`:
  
  <img src="https://github.com/ShirFrenkel/K-means-and-Spectral-Clustering/blob/master/results%20examples/2d_5k_150n_noRandom.PNG" width='500px'>
  
clusters.pdf visualization created by `$ python -m invoke run -k=6 -n=250 --no-Random`:
  
  <img src="https://github.com/ShirFrenkel/K-means-and-Spectral-Clustering/blob/master/results%20examples/3d_6k_250n_noRandom.PNG" width='500px'>


## Files and Modules
   * **config.py:** Program's constants.
   * **main.py:** The main module of the program. Handles the input from the user, data generation and glues everything together.
   * **kmeans.c:** CAPI extension of the K-means algorithm implementation.
   * **kmeans_pp.py:** KMeans++ initialization algorithm, and caller to the kmeans CAPI module.
   * **algebra.py:** Linear algebra algorithms and the Spectral clustering algorithm implementation.
   * **output.py:** Results processing and outputting. 
   * **point_cluster_map.py:** A helper class used to make the output writing more flexible.
   * **setup.py:**  Installation of kmeans CAPI extension.
   * **tasks.py:** Used for the invoke tasks.


## Usage
**For running the program, execute the following command:**
`$ python -m invoke run -k={k} -n={n} --[no-]Random`
when {k}, {n} are replaced with the number of clusters and the number of data points respectively.

**Notes:**
- If random (default): the program casts *n*, *k* to generate data and uses *heuristic-k* (using Eigengap Method) to run the clustering algorithms.
otherwise, it generates data and clusters using the given n, k.
- The choice between 2 or 3 dimension will be random.
- Text inside [ ] means it is optional.
	
