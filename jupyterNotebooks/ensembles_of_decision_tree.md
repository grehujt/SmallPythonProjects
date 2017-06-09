# Ensembles of Decision Trees

## Random forests
- bagging + decision trees
- __n_estimators__ parameter of RandomForestRegressor or RandomForestClassifier
- There are two ways in which the trees in a random forest are randomized: 
    + by randomly selecting the data points used to build a tree (bootstrap)
    + by randomly selecting the subset features in each split test 
- Similarly to the decision tree, the random forest provides __feature importances__

### Strengths, weaknesses, and parameters.
- often work well without heavy tuning of the parameters, and don’t require scaling of the data.
- can be parallelized across multiple CPU cores within a computer easily. (n_jobs)
- The more trees there are in the forest, the more robust it will be against the choice of random state.
- __don’t__ tend to perform well on very high dimensional, sparse data, such as text data.
- require more memory and are slower to train and to predict than linear models
- good rule of thumb to use the default values: __max_features=sqrt(n_features)__ for classification and __max_features=log2(n_features)__ for regression. 

## Gradient boosted regression trees (GBRT)
- Despite the “regression” in the name, these models can be used for regression and classification.
- By default, there is __no randomization__ in gradient boosted regression trees; instead, strong pre-pruning is used.
- often use very shallow trees, which makes the model smaller in terms of memory and makes predictions faster.
- Apart from the pre-pruning and the number of trees in the ensemble, another important parameter of gradient boosting is the __learning_rate__, which controls how strongly each tree tries to correct the mistakes of the previous trees. 
- feature importances
- to a large-scale problem, it might be worth looking into the __xgboost__ package

### Strengths, weaknesses, and parameters
- __main drawback__ is that they require careful tuning of the parameters and may take a long time to train
- Similarly to other tree-based models, the algorithm works well without scaling and on a mixture of binary and continuous features
- often does not work well on high-dimensional sparse data
- __n_estimators__ and the __learning_rate__
    + In contrast to random forests, where a higher n_estimators value is always better, increasing n_estimators in gradient boosting leads to a more complex model, which may lead to overfitting.
    + A common practice is to fit n_estimators depending on the time and memory budget, and then search over dif‐ ferent learning_rates.
- max_depth (or alternatively max_leaf_nodes), to reduce the complexity of each tree. Usually max_depth is set very low for gradient boosted models, often not deeper than five splits.
