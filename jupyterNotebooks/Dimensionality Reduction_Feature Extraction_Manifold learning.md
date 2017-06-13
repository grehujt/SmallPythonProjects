# Dimensionality Reduction, Feature Extraction, and Manifold Learning

## Dimensionality Reduction: Principal Component Analysis (PCA)
- Principal component analysis is a method that rotates the dataset in a way such that the rotated features are statistically uncorrelated.
- One of the most common applications of PCA is visualizing high-dimensional datasets.
- Another application of PCA is feature extraction. It is possible to find a representation of your data that is better suited to analysis than the raw representation.


## Feature Extraction: Non-Negative Matrix Factorization (NMF)
- It works similarly to PCA and can also be used for dimensionality reduction. 
- this method can only be applied to data where each feature is __non-negative__.
- The process of decomposing data into a non-negative weighted sum is particularly helpful for data that is created as the addition (or overlay) of several independent sources, such as an audio track of multiple people speaking, or music with many instruments. In these situations, NMF can identify the original components that make up the combined data.
- Components in NMF are also not ordered in any specific way, so there is no “first non-negative component”: __all components play an equal part__.
- PCA finds the optimum directions in terms of reconstruction. NMF is usually not used for its ability to reconstruct or encode data, but rather for finding interesting patterns within the data.
- There are many other algorithms that can be used to decompose each data point into a weighted sum of a fixed set of components, as PCA and NMF do:
    + independent component analysis (ICA)
    + factor analysis (FA)
    + sparse coding (dictionary learning)

## Manifold Learning with t-SNE
- Manifold learning algorithms are mainly aimed at __visualization__, and so are rarely used to generate more than two new features.
-  The idea behind t-SNE is to find a two-dimensional representation of the data that preserves the distances between points as best as possible, it tries to preserve the information indicating which points are neighbors to each other.
-  The t-SNE algorithm has some tuning parameters, perplexity and early_exaggeration, though it often works well with the default settings. 
