# Clustering

## K-Means
- The algorithm alternates between two steps: 
    + assigning each data point to the closest cluster center, 
    + and then setting each cluster center as the mean of the data points that are assigned to it.
- Each cluster is defined solely by its center, which means that each cluster is a convex shape. As a result of this, k-means can only capture relatively simple shapes. k-means also assumes that all clusters have the same “diameter” in some sense.
- Also assumes that all directions are equally important for each cluster.
- there are interesting parallels between k-means and the decomposition methods like PCA and NMF.
    + PCA tries to find directions of maximum variance in the data
    + while NMF tries to find additive components, which often correspond to “extremes” or “parts” of the data
    + Both methods tried to express the data points as a sum over some components. k-means, on the other hand, tries to represent each data point using a cluster center.
    + This view of k-means as a decomposition method, where each point is represented using a single component, is called __vector quantization__.
- An interesting aspect of vector quantization using k-means is that we can use many more clusters than input dimensions to encode the data.
- scikit-learn even includes a more scalable variant in the __MiniBatchKMeans__ class, which can handle very large datasets.
- One of the drawbacks of k-means is that it relies on a random initialization, which means the outcome of the algorithm depends on a random seed.

## Agglomerative Clustering (Hierarchical clustering)
- refers to a collection of clustering algorithms that all build upon the same principles: the algorithm starts by declaring each point its own cluster, and then merges the two most similar clusters until some stopping criterion is satisfied.
- The stopping criterion implemented in scikit-learn is the number of clusters.
- There are several linkage criteria that specify how exactly the “most similar cluster” is measured. This measure is always defined between two existing clusters:
    + ward, the default choice, ward picks the two clusters to merge such that the variance within all clusters increases the least. This often leads to clusters that are __relatively equally sized__.
    + average, average linkage merges the two clusters that have the smallest average distance between all their points.
    + complete, complete linkage (also known as maximum linkage) merges the two clusters that have the smallest maximum distance between their points.
- If the clusters have very dissimilar numbers of members, average or complete might work better.
- SciPy provides a function that takes a data array X and computes a linkage array, which encodes hierarchical cluster similarities. We can then feed this linkage array into the scipy __dendrogram__ function to plot the dendrogram.

## DBSCAN (density-based spatial clustering of applications with noise)
- can capture clusters of complex shapes, and it can identify points that are not part of any cluster.
- DBSCAN is somewhat slower than agglomerative clustering and k-means, but still scales to relatively large datasets.
- There are two parameters in DBSCAN: __min_samples__ and __eps__, at least min_samples many data points within a distance of eps to a given data point, that data point is classified as a core sample.
- Finding a good setting for eps is sometimes easier after __scaling__ the data using StandardScaler or MinMaxScaler, as using these scaling techniques will ensure that all features have similar ranges.

## Comparing and Evaluating Clustering Algorithms
- sklearn.metrics.cluster
- Evaluating clustering with ground truth
    + adjusted rand index (ARI)
    + normalized mutual information (NMI)
- Evaluating clustering without ground truth
    + silhouette coeffcient, computes the compactness of a cluster, where higher is better, with a perfect score of 1. While compact clusters are good, compactness doesn’t allow for complex shapes.
    + robustness-based clustering metrics, run an algorithm after adding some noise to the data, or using different parameter settings, and compare the outcomes.
