# Kernelized SVM
- kernel trick allows us to learn a classifier in a higher-dimensional space without actually computing the new, possibly very large representation.
- the polynomial kernel, which computes all possible polynomials up to a certain degree of the original features
- the radial basis function (RBF) kernel, also known as the Gaussian kernel, corresponds to an infinite-dimensional feature space, it considers all possible polynomials of all degrees, but the importance of the features decreases for higher degrees.
- the importance of the support vectors that was learned during training (stored in the dual_coef_ attribute of SVC)
- The __gamma__ parameter, bandwidth of Gaussian, is the one shown in the formula given in the previous section, which controls the width of the Gaussian kernel.
- The C parameter is a regularization parameter, similar to that used in the linear models.
- a high value of gamma yields a more complex model.
- a small C means a very restricted model
- A common rescaling method for kernel SVMs is to scale the data such that all features are between 0 and 1. (__MinMaxScaler__)

## Strengths, weaknesses, and parameters
- work well on low-dimensional and high-dimensional data
- __don’t scale very well__ with the number of samples (up to 10,000 samples)
- require careful __preprocessing__ of the data and tuning of the parameters
- hard to inspect
- gamma and C both control the complexity of the model, with large values in either resulting in a more complex model. For the two parameters are usually strongly correlated, and C and gamma should be adjusted together.

