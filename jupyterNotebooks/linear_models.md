# Regression

## Linear Regression
- Linear regression, or ordinary least squares (OLS), is the simplest and most classic linear method for regression.
- finds the parameters w and b that minimize the mean squared error between predictions and the true regression targets, y, on the training set.
- has no parameters, which is a benefit, but it also has no way to control model complexity.

## Ridge Regression (L2 regularization)
- wants the magnitude of coefficients to be as small as possible; in other words, all entries of w should be close to zero.
- Regularization means explicitly restricting a model to avoid overfitting. The particular kind used by ridge regression is known as L2 regularization.
- How much importance the model places on simplicity versus training set performance can be specified by the user, using the alpha parameter.
- ridge is regularized, the training score of ridge is lower than the training score for linear regression.
- test score for ridge is better, particularly for small subsets of the data
- with enough training data, regularization becomes less important, and given enough data, ridge and linear regression will have the same performance.

## Lasso (L1 regularization)
- The consequence of L1 regularization is that when using the lasso, some coefficients are exactly zero. This can be seen as a form of __automatic fea‐ ture selection__.
- also has a regularization parameter, alpha, that controls how strongly coefficients are pushed toward zero.
- __In practice, ridge regression is usually the first choice between these two models__.
- if you have a large amount of features and expect only a few of them to be important, Lasso might be a better choice.

# Linear models for classification

## logistic regression and linear support vector machines
- L2 regularization by default
- the trade-off parameter that determines the strength of the regularization is called C
- __higher values of C correspond to less regularization__
- in high dimensions, linear models for classification become very powerful

## Linear models for multiclass classification
- one-vs.-rest approach, a binary model is learned for each class that tries to separate that class from all of the other classes, resulting in as many binary models.
- To make a prediction, all binary classifiers are run on a test point. The classifier that has the highest score on its single class “wins,” and this class label is returned as the prediction.

# Strengths, weaknesses, and parameters
- The main parameter of linear models is the regularization parameter, called alpha in the regression models and C in LinearSVC and LogisticRegression.
- Large values for alpha or small values for C mean simple models.
- Usually C and alpha are searched for on a logarithmic scale.
- use L1 regularization or L2 regularization
- Linear models are very fast to train, and also fast to predict.
- If your data consists of hundreds of thousands or millions of samples, you might want to investigate using the __solver='sag'__ option in LogisticRegression and Ridge, which can be faster than the default on large datasets.
- __SGDClassifier class and the SGDRegressor__ class, which implement even more scalable versions of the linear models described here.
- Linear models often perform well when the number of features is large compared to the number of samples.
- They are also often used on very large datasets, simply because it’s not feasible to train other models.
- However, in lower-dimensional spaces, other models might yield better generalization performance.
