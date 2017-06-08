# Decision tress

- DecisionTreeRegressor and DecisionTreeClassifier
- Controlling complexity of decision trees
    + pre-pruning (implemented in sklearn)
        * limiting the maximum depth of the tree
        * limiting the maximum number of leaves
        * requiring a minimum number of points in a node to keep splitting it
    + post-pruning
- The most commonly used summary is __feature importance__, which rates how important each feature is for the decision a tree makes. (**tree.feature_importances_**)

```python
def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center') 
    plt.yticks(np.arange(n_features), cancer.feature_names) 
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")


plot_feature_importances_cancer(tree)
```

- The DecisionTreeRegressor (and all other tree-based regression models) is __not able to extrapolate__, or make predictions outside of the range of the training data.

## Strengths, weaknesses, and parameters
- the parameters that control model complexity in decision trees are the pre-pruning parameters that stop the building of the tree before it is fully developed
- the resulting model can easily be visualized and understood
- completely invariant to scaling of the data
- no preprocessing like normalization or standardization of features is needed
- work well when you have features that are on completely different scales, or a mix of binary and continuous features
- __downside__ of decision trees is that even with the use of pre-pruning, they tend to overfit
