# Representing Data and Engineering Features

## Categorical Variables
- one-hot-encoding (or one-out-of-N encoding), which is to replace a categorical variable with one or more new features that can have the values 0 and 1.
- There is a very simple way to encode the data in pandas, using the __get_dummies__ function. The get_dummies function automatically transforms all columns that have object type (like strings) or are categorical into one-hot-encoding.
- This is important to ensure categorical values are represented in the same way in the training set and the test set.
- Be careful: column indexing in pandas includes the end of the range. This is different from slicing a NumPy array, where the end of a range is not included.
- The get_dummies function in pandas treats all numbers as continuous and will not create dummy variables for them. To get around this, you can either use scikit-learn’s __OneHotEncoder__, for which you can specify which variables are continuous and which are discrete, or convert numeric columns in the DataFrame to strings.


## Binning, Discretization for Linear Models and Trees
- The best way to represent data depends not only on the semantics of the data, but also on the kind of model you are using. 
- Binning features generally has no ben‐ eficial effect for tree-based models, as these models can learn to split up the data any‐ where. In a sense, that means decision trees can learn whatever binning is most useful for predicting on this data.
- __np.digitize__
- The OneHotEncoder does the same encoding as pandas.get_dummies, though it currently only works on categorical variables that are integers.
- Binning features generally has __no beneficial effect for tree-based models__, as these models can learn to split up the data any‐ where. In a sense, that means decision trees can learn whatever binning is most useful for predicting on this data.
- the linear model __benefited greatly__ in expressiveness from the binning  transformation of the data.

## Interactions and Polynomials
- If there are good reasons to use a linear model for a particular dataset, say, because it is very large and high-dimensional, but some features have nonlinear relations with the output, binning can be a great way to increase modeling power.
- Using polynomial features together with a linear regression model yields the classical model of polynomial regression, PolynomialFeatures.
- Using a more complex model, a kernel SVM, is able to learn a similarly complex prediction to the polynomial regression without an explicit transformation of the features.
- he interactions and polynomial features gave us a good boost in perfor‐ mance when using Ridge. Without additional features, the random forest beats the performance of Ridge. Adding interactions and polynomials actually decreases random forest performance slightly.

## Univariate Nonlinear Transformations
- linear models and neural networks are very tied to the scale and distribution of each feature, and if there is a nonlinear relation between the feature and the target, that becomes hard to model, particularly in regression.
- applying mathematical functions like log, exp, or sin can help by adjusting the rel‐ ative scales in the data so that they can be captured better by a linear model or neural network.
- Most models work best when each feature (and in regression also the target) is loosely Gaussian distributed, that is, a histogram of each feature should have something resembling the familiar “bell curve” shape. Using transformations like log and exp is a hacky but simple and efficient way to achieve this.
- Finding the transformation that works best for each combination of dataset and model is somewhat of an art. Usually only a subset of the features should be transformed, or sometimes each feature needs to be transformed in a different way.
- these kinds of transformations are irrelevant for tree-based models but might be essential for linear models.
- binning, polynomials, and interactions can have a huge influence on how models perform on a given dataset. This is particularly true for less complex models like linear models and naive Bayes models. Tree-based models, on the other hand, are often able to discover important interactions themselves, and don’t require transforming the data explicitly most of the time. Other models, like SVMs, nearest neighbors, and neural networks, might sometimes benefit from using binning, interactions, or polynomials, but the implications there are usually much less clear than in the case of linear models.

## Automatic Feature Selection
- Univariate Statistics
    + compute whether there is a statistically significant relationship between each feature and the target. Then the features that are related with the highest confidence are selected.
    + A key property of these tests is that they are univariate, meaning that they only consider each feature __individually__.
    + SelectKBest
    + SelectPercentile
- Model-Based Feature Selection
    + uses a supervised machine learning model to judge the importance of each feature, and keeps only the most important ones.
    + The supervised model that is used for feature selection doesn’t need to be the same model that is used for the final supervised modeling.
    + Decision trees and decision tree–based models provide a **feature_importances_** attribute, which directly encodes the importance of each feature.
    + Linear models have __coefficients__, which can also be used to capture feature importances by considering the absolute values.
    + In contrast to univariate selection, model-based selection considers all features at once, and so can capture interactions (if the model can capture them).
- Iterative Feature Selection
    + In univariate testing we used no model, while in model-based selection we used a single model to select features. In iterative feature selection, a series of models are built, with varying numbers of features.
    + There are two basic methods: 
        * starting with no features and adding features one by one until some stopping criterion is reached, one particular method of this kind is __recursive feature elimination (RFE)__, which starts with all features, builds a model, and discards the least important feature according to the model. Then a new model is built using all but the discarded feature, and so on until only a prespecified number of features are left.
        * or starting with all features and removing features one by one until some stopping criterion is reached.
    + Because a series of models are built, these methods are much more computationally expensive.

## Summary
- linear models might benefit greatly from generating new features via binning and adding polynomials and interactions
- while more com‐ plex, nonlinear models like random forests and SVMs might be able to learn more complex tasks without explicitly expanding the feature space
- Trees, and therefore random forests, __cannot extrapolate__ to feature ranges outside the training set. The result is that the model simply predicts the target value of the closest point in the training set, which is the last time it observed any data.
- utilizing expert knowledge in creating derived features
