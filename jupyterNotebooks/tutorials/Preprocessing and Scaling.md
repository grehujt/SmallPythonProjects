# Preprocessing and Scaling
- __StandardScaler__ in scikit-learn ensures that for each feature the mean is 0 and the variance is 1, bringing all features to the same magnitude. However, this scaling does not ensure any particular minimum and maximum values for the features.
- __RobustScaler__ works similarly to the StandardScaler in that it ensures statistical properties for each feature that guarantee that they are on the same scale. However, the RobustScaler __uses the median and quartiles__, instead of mean and variance. This makes the RobustScaler ignore data points that are very different from the rest (like measurement errors).
- The __MinMaxScaler__ shifts the data such that all features are exactly between 0 and 1. 
- __Normalizer__ does a very different kind of rescaling. It scales each data point such that the feature vector has a Euclidean length of 1. In other words, it projects a data point on the circle (or sphere, in the case of higher dimensions) with a radius of 1.
- It is important to apply exactly the same transformation to the training set and the test set for the supervised model to work on the test set.
