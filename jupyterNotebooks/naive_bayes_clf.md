# Naive Bayes Classifiers

- Quite similar to the linear models discussed in the previous section. However, they tend to be even faster in training.
- Naive Bayes models are so efficient is that they learn parameters by __looking at each feature individually__ and __collect simple per-class statistics from each feature__.
- Three kinds of naive Bayes classifiers implemented in sklearn:
    + GaussianNB
        * can be applied to any continuous data
    + BernoulliNB
        * assumes binary data
    + MultinomialNB
        * assumes count data (each feature represents an integer count of some‐ thing, like how often a word appears in a sentence)
- BernoulliNB and MultinomialNB are mostly used in __text data classification__.

## Strengths, weaknesses, and parameters
- MultinomialNB and BernoulliNB have a single parameter, __alpha__, which controls model complexity.
- A large alpha means more smoothing, resulting in less complex models.
- The algorithm’s performance is relatively robust to the setting of alpha, meaning that setting alpha is not critical for good performance.
- GaussianNB is mostly used on very high-dimensional data, while the other two var‐ iants of naive Bayes are widely used for sparse count data such as text.
- MultinomialNB usually performs better than BernoulliNB, particularly on datasets with a relatively large number of nonzero features (i.e., large documents).
- The naive Bayes models share many of the strengths and weaknesses of the linear models:
    + very fast to train and to predict, and the training procedure is easy to understand.
    + work very well with high-dimensional sparse data and are relatively robust to the parameters.
- Naive Bayes models are great baseline models and are often used on very large datasets, where training even a linear model might take too long.
